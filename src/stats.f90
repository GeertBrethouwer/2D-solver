!
! statistics computed in spectral space (Parseval), time-averaged wall-normal profiles,
! and the run log
!
module mod_stats
  use mpi
  use mod_types
  use mod_param
  use mod_decomp , only: myid,nkx,kx_lo
  use mod_ysolve , only: wq,mean_owner,i00,wall_gradient
  use mod_compact, only: yg
  implicit none
  private
  public :: init_stats,accumulate_stats,write_stats,log_line,mode_energy,profile_mean
  integer :: nq
  real(rp), allocatable :: ssum(:,:),prof(:)
  real(rp) :: tsum = 0._rp
  integer  :: nsamp = 0
  character(len=12), allocatable :: qname(:)
contains
  subroutine init_stats
    implicit none
    integer :: n
    character(len=4) :: cs
    nq = 5 + 4*nscal
    allocate(ssum(ny,nq),prof(ny),qname(nq))
    ssum(:,:) = 0._rp
    qname(1:5) = [character(len=12) :: 'U','V','uu','vv','uv']
    do n=1,nscal
      write(cs,'(i0)') n
      qname(5+4*(n-1)+1) = 'S'//trim(cs)
      qname(5+4*(n-1)+2) = 's'//trim(cs)//'s'//trim(cs)
      qname(5+4*(n-1)+3) = 'us'//trim(cs)
      qname(5+4*(n-1)+4) = 'vs'//trim(cs)
    end do
    !$acc enter data create(prof)
  end subroutine init_stats
  !
  subroutine correlation(f,g,fac,res)
    !
    ! res(y) = < f' g' >_x = sum over kx > 0 of 2 Re(f g*)
    !
    implicit none
    complex(rp), intent(in) :: f(nkx,ny),g(nkx,ny)
    real(rp), intent(in) :: fac
    real(rp), intent(out) :: res(ny)
    integer :: i,k,ierr
    real(rp) :: acc,wt
    !$acc parallel loop gang default(present) private(acc) async(1)
    do k=1,ny
      acc = 0._rp
      !$acc loop vector reduction(+:acc) private(wt)
      do i=1,nkx
        wt = 2._rp
        if(kx_lo+i-1 == 1) wt = 0._rp
        acc = acc + wt*real(f(i,k)*conjg(g(i,k)),rp)
      end do
      prof(k) = acc*fac
    end do
    !$acc wait(1)
    !$acc update self(prof)
    res(:) = prof(:)
    call MPI_ALLREDUCE(MPI_IN_PLACE,res,ny,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,ierr)
  end subroutine correlation
  !
  subroutine profile_mean(f,fac,res)
    ! mean (kx = 0) profile, available on all ranks
    implicit none
    complex(rp), intent(in) :: f(nkx,ny)
    real(rp), intent(in) :: fac
    real(rp), intent(out) :: res(ny)
    integer :: k,ierr
    if(mean_owner) then
      !$acc parallel loop default(present) async(1)
      do k=1,ny
        prof(k) = real(f(i00,k),rp)*fac
      end do
      !$acc wait(1)
      !$acc update self(prof)
      res(:) = prof(:)
    else
      res(:) = 0._rp
    end if
    call MPI_ALLREDUCE(MPI_IN_PLACE,res,ny,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,ierr)
  end subroutine profile_mean
  !
  subroutine instantaneous(u,v,s,st)
    implicit none
    complex(rp), intent(in) :: u(nkx,ny),v(nkx,ny)
    complex(rp), intent(in), contiguous :: s(:,:,:)
    real(rp), intent(out) :: st(ny,nq)
    integer :: n,o
    call profile_mean(u,1._rp,st(:,1))
    call profile_mean(v,1._rp,st(:,2))
    call correlation(u,u,1._rp,st(:,3))
    call correlation(v,v,1._rp,st(:,4))
    call correlation(u,v,1._rp,st(:,5))
    do n=1,nscal
      o = 5+4*(n-1)
      call profile_mean(s(:,:,n),1._rp,st(:,o+1))
      call correlation(s(:,:,n),s(:,:,n),1._rp,st(:,o+2))
      call correlation(u,s(:,:,n),1._rp,st(:,o+3))
      call correlation(v,s(:,:,n),1._rp,st(:,o+4))
    end do
  end subroutine instantaneous
  !
  subroutine accumulate_stats(u,v,s,dtcur)
    ! time-weighted average
    implicit none
    complex(rp), intent(in), contiguous :: u(:,:),v(:,:),s(:,:,:)
    real(rp), intent(in) :: dtcur
    real(rp) :: st(ny,nq)
    call instantaneous(u,v,s,st)
    ssum(:,:) = ssum(:,:) + dtcur*st(:,:)
    tsum = tsum + dtcur
    nsamp = nsamp + 1
  end subroutine accumulate_stats
  !
  subroutine write_stats(time)
    implicit none
    real(rp), intent(in) :: time
    integer :: iunit,k,q
    character(len=32) :: fmt
    if(myid /= 0 .or. nsamp == 0) return
    open(newunit=iunit,file=trim(data_dir)//'/stats.dat',status='replace')
    write(iunit,'(a,es14.6,a,es14.6,a,i0)') '# time-averaged profiles; t_end = ',time, &
                                            ', averaging time = ',tsum,', samples = ',nsamp
    write(iunit,'(a)') '# fluctuation moments are <f''g''> (average over x)'
    write(iunit,'(a)',advance='no') '#            y'
    do q=1,nq
      write(iunit,'(a16)',advance='no') trim(qname(q))
    end do
    write(iunit,*)
    write(fmt,'(a,i0,a)') '(',nq+1,'es16.7e3)'
    do k=1,ny
      write(iunit,fmt) yg(k),(ssum(k,q)/tsum,q=1,nq)
    end do
    close(iunit)
  end subroutine write_stats
  !
  real(rp) function mode_energy(u,v,ikx)
    ! integral over y of |u|^2+|v|^2 of Fourier mode ikx (>= 0)
    implicit none
    complex(rp), intent(in) :: u(nkx,ny),v(nkx,ny)
    integer, intent(in) :: ikx
    integer :: i,k,ierr
    real(rp) :: e
    e = 0._rp
    i = ikx+1-kx_lo+1
    if(i >= 1 .and. i <= nkx) then
      !$acc parallel loop default(present) reduction(+:e) async(1)
      do k=1,ny
        e = e + wq(k)*(abs(u(i,k))**2+abs(v(i,k))**2)
      end do
      !$acc wait(1)
    end if
    call MPI_ALLREDUCE(MPI_IN_PLACE,e,1,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,ierr)
    mode_energy = e
  end function mode_energy
  !
  subroutine log_line(iunit,istep,time,dtcur,divm,u,v,s,dpdx_app)
    implicit none
    integer, intent(in) :: iunit,istep
    real(rp), intent(in) :: time,dtcur,divm,dpdx_app
    complex(rp), intent(in) :: u(nkx,ny),v(nkx,ny)
    complex(rp), intent(in), contiguous :: s(:,:,:)
    real(rp) :: um(ny),uu(ny),vv(ny),ub,gb,gt,tauw,retau,ekin,em
    real(rp) :: buf(2+2*nscal_max)
    integer :: n,ierr
    call profile_mean(u,1._rp,um)
    call correlation(u,u,1._rp,uu)
    call correlation(v,v,1._rp,vv)
    ub = sum(wq(:)*um(:))/hgt
    ekin = 0.5_rp*sum(wq(:)*(uu(:)+vv(:)))/hgt
    buf(:) = 0._rp
    call wall_gradient(u,buf(1),buf(2))
    do n=1,nscal
      call wall_gradient(s(:,:,n),buf(2+2*n-1),buf(2+2*n))
    end do
    call MPI_ALLREDUCE(MPI_IN_PLACE,buf,2+2*nscal,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,ierr)
    gb = buf(1); gt = buf(2)
    if(freeslip_bot) then
      tauw = visc*abs(gt)                         ! free-slip wall y = 0: shear stress at y = ly
    else if(freeslip_top) then
      tauw = visc*abs(gb)                         ! free-slip wall y = ly: shear stress at y = 0
    else
      tauw = 0.5_rp*visc*(abs(gb)+abs(gt))
    end if
    ! Re_tau = u_tau delta/nu with delta = half the wall distance (closed channel) or the full
    ! wall distance (one free-slip wall: open channel)
    retau = sqrt(tauw)*merge(hgt,hhalf,freeslip_bot.or.freeslip_top)/visc
    em = 0._rp
    if(mode_log_kx >= 0) em = mode_energy(u,v,mode_log_kx)
    if(myid == 0) then
      write(iunit,'(i9,6es14.6,es11.3,2es14.6)',advance='no') istep,time,dtcur,ub,dpdx_app,tauw,retau,divm,ekin,em
      do n=1,nscal
        write(iunit,'(2es14.6)',advance='no') buf(2+2*n-1),buf(2+2*n)
      end do
      write(iunit,*)
      flush(iunit)
    end if
  end subroutine log_line
end module mod_stats
