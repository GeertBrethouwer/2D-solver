!
! transforms between the physical (dealiased) x-lines and the spectral y-lines
!
!   physical : real    f (mx , nyA)    x in [0,lx) on the (3/2-)padded grid, y-rows of this rank
!   spectral : complex fh(nkx, ny )    kx = 0..nx/2-1 (the Nyquist mode is dropped), all y
!
!   f(x,y) = sum_kx fh(kx,y) exp(i kx x),  kx >= 0 with Hermitian symmetry
!
! GPU: batched cuFFT plans on the OpenACC stream (as in HiCHAN/CaNS); CPU: FFTW3.
!
module mod_fft
  use, intrinsic :: iso_c_binding
  use mod_types
  use mod_param , only: nxh,ny,mx,abort_run
  use mod_decomp, only: nyA,nkx,tr_x2y,tr_y2x
  use mod_timer , only: tic,toc,tm_fft,tm_tr
#if defined(_CUDA)
  use cufft
  use mod_decomp, only: istream
#endif
  implicit none
#if !defined(_CUDA)
  include 'fftw3.f03'
#endif
  private
  public :: init_fft,fwd_fft,bwd_fft,end_fft
  complex(rp), allocatable :: cxw(:,:),xaw(:,:)   ! (mx/2+1,nyA) FFT output, (nxh,nyA) kept modes
#if defined(_CUDA)
  integer :: plan_xf,plan_xb
#else
  type(c_ptr) :: plan_xf,plan_xb
#endif
  real(rp) :: norm
contains
  subroutine init_fft(phys)
    implicit none
    real(rp), intent(inout) :: phys(mx,nyA) ! only used for planning (not modified)
#if defined(_CUDA)
    integer :: istat
#endif
    norm = 1._rp/real(mx,rp)
    allocate(cxw(mx/2+1,nyA),xaw(nxh,nyA))
    !$acc enter data create(cxw,xaw)
#if defined(_CUDA)
    istat = cufftPlanMany(plan_xf,1,[mx],[mx],1,mx,[mx/2+1],1,mx/2+1,CUFFT_D2Z,nyA)
    if(istat /= CUFFT_SUCCESS) call abort_run('cuFFT plan (x, forward)')
    istat = cufftPlanMany(plan_xb,1,[mx],[mx/2+1],1,mx/2+1,[mx],1,mx,CUFFT_Z2D,nyA)
    if(istat /= CUFFT_SUCCESS) call abort_run('cuFFT plan (x, backward)')
    istat = cufftSetStream(plan_xf,istream)
    istat = cufftSetStream(plan_xb,istream)
#else
    plan_xf = fftw_plan_many_dft_r2c(1,[mx],nyA,phys,[mx],1,mx,cxw,[mx/2+1],1,mx/2+1,FFTW_ESTIMATE)
    plan_xb = fftw_plan_many_dft_c2r(1,[mx],nyA,cxw,[mx/2+1],1,mx/2+1,phys,[mx],1,mx,FFTW_ESTIMATE)
    if(.not.(c_associated(plan_xf).and.c_associated(plan_xb))) call abort_run('FFTW planning failed')
#endif
  end subroutine init_fft
  !
  subroutine end_fft
    implicit none
#if defined(_CUDA)
    integer :: istat
    istat = cufftDestroy(plan_xf); istat = cufftDestroy(plan_xb)
#else
    call fftw_destroy_plan(plan_xf); call fftw_destroy_plan(plan_xb)
#endif
    !$acc exit data delete(cxw,xaw)
  end subroutine end_fft
  !
  subroutine fwd_fft(phys,fh)
    ! physical -> spectral; `phys` may be overwritten
    implicit none
    real(rp)   , intent(inout) :: phys(mx,nyA)
    complex(rp), intent(out)   :: fh(nkx,ny)
    integer :: i,k
#if defined(_CUDA)
    integer :: istat
#endif
    call tic(tm_fft)
#if defined(_CUDA)
    !$acc host_data use_device(phys,cxw)
    istat = cufftExecD2Z(plan_xf,phys,cxw)
    !$acc end host_data
#else
    call fftw_execute_dft_r2c(plan_xf,phys,cxw)
#endif
    !$acc parallel loop collapse(2) default(present) async(1)
    do k=1,nyA
      do i=1,nxh
        xaw(i,k) = cxw(i,k)*norm
      end do
    end do
    call tic(tm_tr); call tr_x2y(xaw,fh); call toc(tm_tr)
    call toc(tm_fft)
  end subroutine fwd_fft
  !
  subroutine bwd_fft(fh,phys)
    ! spectral -> physical, with zero padding
    implicit none
    complex(rp), intent(in)  :: fh(nkx,ny)
    real(rp)   , intent(out) :: phys(mx,nyA)
    integer :: i,k
#if defined(_CUDA)
    integer :: istat
#endif
    call tic(tm_fft)
    call tic(tm_tr); call tr_y2x(fh,xaw); call toc(tm_tr)
    !$acc parallel loop collapse(2) default(present) async(1)
    do k=1,nyA
      do i=1,mx/2+1
        if(i <= nxh) then
          cxw(i,k) = xaw(i,k)
        else
          cxw(i,k) = (0._rp,0._rp)
        end if
      end do
    end do
#if defined(_CUDA)
    !$acc host_data use_device(phys,cxw)
    istat = cufftExecZ2D(plan_xb,cxw,phys)
    !$acc end host_data
#else
    call fftw_execute_dft_c2r(plan_xb,cxw,phys)
#endif
    call toc(tm_fft)
  end subroutine bwd_fft
end module mod_fft
