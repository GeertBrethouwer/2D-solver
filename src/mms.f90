!
! method of manufactured solutions (MMS) with the melt condition: &mms_test, mms = .true.
!
! The manufactured solution (tools/mms/mms_solution.py -> src/mms_solution.f90) satisfies all boundary
! conditions of the ice set-up exactly: the three-equation melt condition, u = 0 and v = m* at y = 0, and a
! free-slip wall with fixed scalar values at y = ly. Source terms f (momentum) and Q (scalars) make it an
! exact solution of the equations. The code
!   * starts from the exact solution at t = 0 (unless it restarts),
!   * adds the sources to the explicit (RK3) terms at the stage times t_q = t + c_q dt, c = 0, 8/15, 2/3,
!   * writes the error to data_dir/mms_error.dat at every log step: max over the (dealiased) physical
!     grid of |u - u_ex|, |v - v_ex|, |theta - theta_ex|, |s - s_ex|, and at the ice (y = 0) the errors of
!     the melt rate m* (= v there), theta_b and s_b.
! Requirements (checked in mod_param): nscal = 2, melt with melt_t = 1, melt_s = 2, melt_blowing,
! bc_bot = 'noslip', bc_top = 'freeslip', forcing = 'cpg' with dpdx = 0, Dirichlet scalar values at y = ly.
!
module mod_mms
  use mpi
  use mod_types
  use mod_param
  use mod_decomp , only: myid,nkx,nyA,yA_lo
  use mod_compact, only: yg
  use mod_fft    , only: fwd_fft,bwd_fft
  use mod_mms_solution, only: nprm,mms_exact,mms_source,mms_interface
  implicit none
  private
  public :: init_mms,mms_init_fields,mms_add_sources,mms_error
  real(rp) :: prm(nprm)
  real(rp), allocatable :: ygd(:)           ! y of the rows of this rank
  real(rp), allocatable :: wp(:,:,:)        ! (mx,nyA,4) physical work
  complex(rp), allocatable :: wh(:,:)       ! (nkx,ny) spectral work
  integer :: iunit = -1
contains
  subroutine init_mms
    implicit none
    integer :: k
    prm = [visc,diff(1),diff(2),ri(1),ri(2),gvec(1),gvec(2),melt_st,melt_lambda,melt_thf,melt_sigma, &
           lx,ly,sval_top(1),sval_top(2)]
    allocate(ygd(nyA),wp(mx,nyA,4),wh(nkx,ny))
    do k=1,nyA
      ygd(k) = yg(yA_lo+k-1)
    end do
    wp = 0; wh = 0
    !$acc enter data copyin(prm,ygd,wp,wh)
    if(myid == 0) then
      open(newunit=iunit,file=trim(data_dir)//'/mms_error.dat',position='append')
      write(iunit,'(a)') '#           time            e_u            e_v        e_theta            e_s'// &
                         '    e_m*(wall)  e_theta_b(wall)    e_s_b(wall)'
    end if
  end subroutine init_mms
  !
  subroutine mms_init_fields(u,v,s)
    ! u, v, theta, s from the exact solution at t = 0
    implicit none
    complex(rp), intent(inout) :: u(nkx,ny),v(nkx,ny),s(nkx,ny,max(nscal,1))
    integer :: i,k
    real(rp) :: x,ue,ve,te,se
    !$acc parallel loop collapse(2) default(present) private(x,ue,ve,te,se) async(1)
    do k=1,nyA
      do i=1,mx
        x = lx*(i-1)/real(mx,rp)
        call mms_exact(x,ygd(k),0._rp,prm,ue,ve,te,se)
        wp(i,k,1) = ue; wp(i,k,2) = ve; wp(i,k,3) = te; wp(i,k,4) = se
      end do
    end do
    call fwd_fft(wp(:,:,1),u)
    call fwd_fft(wp(:,:,2),v)
    call fwd_fft(wp(:,:,3),s(:,:,1))
    call fwd_fft(wp(:,:,4),s(:,:,2))
    !$acc wait(1)
  end subroutine mms_init_fields
  !
  subroutine mms_add_sources(tq,nxf,nyf,ns)
    ! add the source terms at time tq to the explicit terms N_x, N_y (nxf, nyf) and N of the scalars
    implicit none
    real(rp), intent(in) :: tq
    complex(rp), intent(inout) :: nxf(nkx,ny),nyf(nkx,ny),ns(nkx,ny,max(nscal,1))
    integer :: i,k,m
    real(rp) :: x,fx,fy,qt,qs
    !$acc parallel loop collapse(2) default(present) private(x,fx,fy,qt,qs) async(1)
    do k=1,nyA
      do i=1,mx
        x = lx*(i-1)/real(mx,rp)
        call mms_source(x,ygd(k),tq,prm,fx,fy,qt,qs)
        wp(i,k,1) = fx; wp(i,k,2) = fy; wp(i,k,3) = qt; wp(i,k,4) = qs
      end do
    end do
    do m=1,4
      call fwd_fft(wp(:,:,m),wh)
      select case(m)
      case(1)
        call addto(nxf,wh)
      case(2)
        call addto(nyf,wh)
      case(3)
        call addto(ns(:,:,1),wh)
      case(4)
        call addto(ns(:,:,2),wh)
      end select
    end do
  end subroutine mms_add_sources
  !
  subroutine addto(f,g)
    implicit none
    complex(rp), intent(inout) :: f(nkx,ny)
    complex(rp), intent(in) :: g(nkx,ny)
    integer :: i,k
    !$acc parallel loop collapse(2) default(present) async(1)
    do k=1,ny
      do i=1,nkx
        f(i,k) = f(i,k) + g(i,k)
      end do
    end do
  end subroutine addto
  !
  subroutine mms_error(time,u,v,s)
    ! max errors against the exact solution at time `time`, written to mms_error.dat
    implicit none
    real(rp), intent(in) :: time
    complex(rp), intent(in) :: u(nkx,ny),v(nkx,ny),s(nkx,ny,max(nscal,1))
    integer :: i,k,ierr
    real(rp) :: x,ue,ve,te,se,me,tbe,sbe,e(7)
    real(rp) :: e1,e2,e3,e4,e5,e6,e7
    call bwd_fft(u,wp(:,:,1))
    call bwd_fft(v,wp(:,:,2))
    call bwd_fft(s(:,:,1),wp(:,:,3))
    call bwd_fft(s(:,:,2),wp(:,:,4))
    e1 = 0; e2 = 0; e3 = 0; e4 = 0; e5 = 0; e6 = 0; e7 = 0
    !$acc parallel loop collapse(2) default(present) private(x,ue,ve,te,se,me,tbe,sbe) &
    !$acc reduction(max:e1,e2,e3,e4,e5,e6,e7) async(1)
    do k=1,nyA
      do i=1,mx
        x = lx*(i-1)/real(mx,rp)
        call mms_exact(x,ygd(k),time,prm,ue,ve,te,se)
        e1 = max(e1,abs(wp(i,k,1)-ue)); e2 = max(e2,abs(wp(i,k,2)-ve))
        e3 = max(e3,abs(wp(i,k,3)-te)); e4 = max(e4,abs(wp(i,k,4)-se))
        if(yA_lo == 1 .and. k == 1) then
          call mms_interface(x,ygd(k),time,prm,me,tbe,sbe)
          e5 = max(e5,abs(wp(i,k,2)-me)); e6 = max(e6,abs(wp(i,k,3)-tbe)); e7 = max(e7,abs(wp(i,k,4)-sbe))
        end if
      end do
    end do
    !$acc wait(1)
    e = [e1,e2,e3,e4,e5,e6,e7]
    call MPI_ALLREDUCE(MPI_IN_PLACE,e,7,MPI_DOUBLE_PRECISION,MPI_MAX,MPI_COMM_WORLD,ierr)
    if(myid == 0) then
      write(iunit,'(8es15.6e3)') time,e
      flush(iunit)
    end if
  end subroutine mms_error
end module mod_mms
