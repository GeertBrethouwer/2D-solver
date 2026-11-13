!
! initial conditions (set on the host, then copied to the device)
!
! Only v (kx /= 0), the mean flow U and the scalars are free: u (kx /= 0) follows from
! continuity, u = i D1 v/kx. Disturbances of v are built with the factor (1-y^2)^2, so that
! v = dv/dy = 0 at the walls.
!
module mod_initflow
  use mod_types
  use mod_param
  use mod_decomp , only: myid,nkx,kx_lo
  use mod_compact, only: yg
  implicit none
  private
  public :: init_fields
contains
  pure real(rp) function hash01(a,b,c,d)
    ! deterministic pseudo-random number in [0,1) from four integers (decomposition independent)
    implicit none
    integer, intent(in) :: a,b,c,d
    integer(i8) :: h
    h = 1469598103_i8
    h = ieor(h*1099511_i8, int(a,i8)+7919_i8)
    h = ieor(h*1099511_i8, int(b,i8)+104729_i8)
    h = ieor(h*1099511_i8, int(c,i8)+1299709_i8)
    h = ieor(h*1099511_i8, int(d,i8)+15485863_i8)
    h = ieor(h,ishft(h,-29)); h = h*2654435761_i8; h = ieor(h,ishft(h,-32))
    hash01 = real(modulo(h,2147483647_i8),rp)/2147483647._rp
  end function hash01
  !
  complex(rp) function rnd_profile(mx_,k,comp,seed)
    ! smooth random wall-normal profile of mode mx_, zero at the walls
    implicit none
    integer, intent(in) :: mx_,k,comp,seed
    integer :: q
    real(rp) :: y,ph(3),am(3)
    do q=1,3
      am(q) = hash01(mx_,0,comp,seed+10*q) - 0.5_rp
      ph(q) = 2*pi*hash01(mx_,0,comp,seed+10*q+5)
    end do
    y = (yg(k)-ymid)/hhalf                       ! in [-1,1]
    rnd_profile = (1._rp-y*y)*(am(1)*exp(cmplx(0._rp,ph(1),rp)) + am(2)*y*exp(cmplx(0._rp,ph(2),rp)) + &
                               am(3)*(2*y*y-1._rp)*exp(cmplx(0._rp,ph(3),rp)))
  end function rnd_profile
  !
  subroutine init_fields(u,v,s)
    use mod_ysolve, only: vel_from_v
    implicit none
    complex(rp), intent(inout) :: u(nkx,ny),v(nkx,ny)
    complex(rp), intent(inout), contiguous :: s(:,:,:)
    integer :: i,k,n,mxx,iunit,ios,mmax
    real(rp) :: y,ulam,e,fac,uref
    real(rp) :: ur,ui,vr,vi
    complex(rp), allocatable :: mode(:)
    u(:,:) = 0; v(:,:) = 0; s(:,:,:) = 0
    !
    ! laminar profile in the mean mode: Poiseuille U = Umax (1 - eta^2), eta = (y-ymid)/(h/2), or with a
    ! free-slip wall the open-channel profile U = Umax (1 - zeta^2), zeta = distance from the free-slip
    ! wall / h (dU/dy = 0 there, U = 0 at the no-slip wall); both have U_bulk = 2/3 Umax (h = ly)
    !
    if(forcing == 'cfr') then
      ulam = 1.5_rp*ubulk
    else if(freeslip_bot .or. freeslip_top) then
      ulam = 0.5_rp*dpdx*re*hgt**2
    else
      ulam = 0.5_rp*dpdx*re*hhalf**2
    end if
    if(kx_lo == 1) then
      do k=1,ny
        if(freeslip_bot) then
          u(1,k) = ulam*(1._rp-((yg(k)-ybot)/hgt)**2)
        else if(freeslip_top) then
          u(1,k) = ulam*(1._rp-((ytop-yg(k))/hgt)**2)
        else
          u(1,k) = ulam*(1._rp-((yg(k)-ymid)/hhalf)**2)
        end if
      end do
    end if
    uref = (2._rp/3._rp)*abs(ulam)
    if(uref == 0._rp) uref = 1._rp
    !
    select case(trim(init_type))
    case('laminar','restart')
    case('noise')
      ! random v in the modes 1..nx/6, amplitude decreasing with kx; rms of (u',v') = init_amp*U_b
      ! (U_b = laminar bulk velocity, or 1 without mean flow)
      mmax = max(1,nx/6)
      do i=1,nkx
        mxx = kx_lo+i-2
        if(mxx < 1 .or. mxx > mmax) cycle
        do k=2,ny-1
          y = (yg(k)-ymid)/hhalf
          v(i,k) = (1._rp-y*y)*rnd_profile(mxx,k,2,init_seed)/(1._rp+0.2_rp*mxx)
        end do
      end do
    case('mode')
      !
      ! a single Fourier mode (e.g. an Orr-Sommerfeld eigenfunction) read from mode_file:
      ! ny lines with Re(u) Im(u) Re(v) Im(v) [Re(w) Im(w)] (u and w are not used); amplitude init_amp
      !
      allocate(mode(ny))
      open(newunit=iunit,file=trim(mode_file),status='old',action='read',iostat=ios)
      if(ios /= 0) call abort_run('cannot open mode file '//trim(mode_file))
      do k=1,ny
        read(iunit,*,iostat=ios) ur,ui,vr,vi
        if(ios /= 0) call abort_run('mode file too short')
        mode(k) = cmplx(vr,vi,rp)
      end do
      close(iunit)
      if(mode_kx < 1 .or. mode_kx >= nxh) call abort_run('mode_kx must be in 1..nx/2-1')
      i = mode_kx+1-kx_lo+1
      if(i >= 1 .and. i <= nkx) v(i,:) = init_amp*mode(:)
      if(i >= 1 .and. i <= nkx) then
        v(i,1) = 0; v(i,ny) = 0
      end if
    case default
      call abort_run('unknown init_type '//trim(init_type))
    end select
    !
    ! scalars
    !
    do n=1,nscal
      if(kx_lo == 1) then
        do k=1,ny
          y = (yg(k)-ymid)/hhalf
          select case(trim(sinit(n)))
          case('linear','noise')
            if(sbc_bot(n) == 'D' .and. sbc_top(n) == 'D') then
              s(1,k,n) = 0.5_rp*(sval_bot(n)*(1._rp-y) + sval_top(n)*(1._rp+y))
            end if
          case('zero')
          case('one')
            s(1,k,n) = 1._rp                   ! uniform value 1, walls included
          case default
            call abort_run('unknown sinit')
          end select
        end do
      end if
      if(trim(sinit(n)) == 'noise') then
        do i=1,nkx
          mxx = kx_lo+i-2
          if(mxx < 1 .or. mxx > max(1,nx/6)) cycle
          do k=2,ny-1
            s(i,k,n) = s(i,k,n) + init_amp*rnd_profile(mxx,k,3+n,init_seed)
          end do
        end do
      end if
    end do
    !$acc update device(u,v,s)
    !
    ! u from continuity
    !
    call vel_from_v(u,v)
    !$acc wait(1)
    !$acc update self(u,v)
    if(trim(init_type) == 'noise') then
      call fluct_energy(u,v,e)
      fac = init_amp*uref/sqrt(max(e,tiny(1._rp)))
      do i=1,nkx
        if(kx_lo+i-1 == 1) cycle
        u(i,:) = fac*u(i,:); v(i,:) = fac*v(i,:)
      end do
      !$acc update device(u,v)
    end if
  end subroutine init_fields
  !
  subroutine fluct_energy(u,v,e)
    ! e = < u'u' + v'v' >/2 (volume average, Parseval)
    use mpi
    use mod_compact, only: wq
    implicit none
    complex(rp), intent(in) :: u(nkx,ny),v(nkx,ny)
    real(rp), intent(out) :: e
    integer :: i,k,ierr
    real(rp) :: wt
    e = 0._rp
    do k=1,ny
      do i=1,nkx
        wt = 2._rp
        if(kx_lo+i-1 == 1) wt = 0._rp
        e = e + wq(k)*wt*(abs(u(i,k))**2+abs(v(i,k))**2)/(2._rp*hgt)
      end do
    end do
    call MPI_ALLREDUCE(MPI_IN_PLACE,e,1,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,ierr)
  end subroutine fluct_energy
end module mod_initflow
