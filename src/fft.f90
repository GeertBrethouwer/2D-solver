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
  use mod_decomp, only: nyA,nkx,tr_x2y,tr_y2x,row_gather,row_scatter,myid
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
  public :: init_fft,fwd_fft,bwd_fft,end_fft,row_bwd,row_fwd
  complex(rp), allocatable :: cxw(:,:),xaw(:,:)   ! (mx/2+1,nyA) FFT output, (nxh,nyA) kept modes
  ! transforms of a single row (the wall row of the melt model): the spectral row is gathered on
  ! rank 0, which holds the physical wall row
  complex(rp), allocatable :: rspec(:),rglob(:)   ! (mx/2+1) FFT input/output, (nxh) the gathered row
  real(rp)   , allocatable :: rphys(:)            ! (mx) only used for planning
#if defined(_CUDA)
  integer :: plan_xf,plan_xb,plan_rf,plan_rb
#else
  type(c_ptr) :: plan_xf,plan_xb,plan_rf,plan_rb
  integer(c_int) :: flags
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
    allocate(cxw(mx/2+1,nyA),xaw(nxh,nyA),rspec(mx/2+1),rglob(nxh),rphys(mx))
    rspec = 0; rglob = 0; rphys = 0
    !$acc enter data create(cxw,xaw) copyin(rspec,rphys)
#if defined(_CUDA)
    istat = cufftPlanMany(plan_xf,1,[mx],[mx],1,mx,[mx/2+1],1,mx/2+1,CUFFT_D2Z,nyA)
    if(istat /= CUFFT_SUCCESS) call abort_run('cuFFT plan (x, forward)')
    istat = cufftPlanMany(plan_xb,1,[mx],[mx/2+1],1,mx/2+1,[mx],1,mx,CUFFT_Z2D,nyA)
    if(istat /= CUFFT_SUCCESS) call abort_run('cuFFT plan (x, backward)')
    istat = cufftPlanMany(plan_rf,1,[mx],[mx],1,mx,[mx/2+1],1,mx/2+1,CUFFT_D2Z,1)
    if(istat /= CUFFT_SUCCESS) call abort_run('cuFFT plan (row, forward)')
    istat = cufftPlanMany(plan_rb,1,[mx],[mx/2+1],1,mx/2+1,[mx],1,mx,CUFFT_Z2D,1)
    if(istat /= CUFFT_SUCCESS) call abort_run('cuFFT plan (row, backward)')
    istat = cufftSetStream(plan_xf,istream)
    istat = cufftSetStream(plan_xb,istream)
    istat = cufftSetStream(plan_rf,istream)
    istat = cufftSetStream(plan_rb,istream)
#else
    ! the field plans are also executed on slices of work arrays, e.g. gp(:,:,2), which start mx*nyA
    ! reals after an aligned address: not 16-byte aligned if that number is odd
    flags = FFTW_ESTIMATE
    if(mod(mx,2) == 1 .and. mod(nyA,2) == 1) flags = ior(flags,FFTW_UNALIGNED)
    plan_xf = fftw_plan_many_dft_r2c(1,[mx],nyA,phys,[mx],1,mx,cxw,[mx/2+1],1,mx/2+1,flags)
    plan_xb = fftw_plan_many_dft_c2r(1,[mx],nyA,cxw,[mx/2+1],1,mx/2+1,phys,[mx],1,mx,flags)
    ! the row plans are executed on columns of small arrays
    flags = ior(FFTW_ESTIMATE,FFTW_UNALIGNED)
    plan_rf = fftw_plan_many_dft_r2c(1,[mx],1,rphys,[mx],1,mx,rspec,[mx/2+1],1,mx/2+1,flags)
    plan_rb = fftw_plan_many_dft_c2r(1,[mx],1,rspec,[mx/2+1],1,mx/2+1,rphys,[mx],1,mx,flags)
    if(.not.(c_associated(plan_xf).and.c_associated(plan_xb).and.c_associated(plan_rf).and.c_associated(plan_rb))) &
      call abort_run('FFTW planning failed')
#endif
  end subroutine init_fft
  !
  subroutine end_fft
    implicit none
#if defined(_CUDA)
    integer :: istat
    istat = cufftDestroy(plan_xf); istat = cufftDestroy(plan_xb)
    istat = cufftDestroy(plan_rf); istat = cufftDestroy(plan_rb)
#else
    call fftw_destroy_plan(plan_xf); call fftw_destroy_plan(plan_xb)
    call fftw_destroy_plan(plan_rf); call fftw_destroy_plan(plan_rb)
#endif
    !$acc exit data delete(cxw,xaw,rspec,rphys)
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
  !
  subroutine row_bwd(h,p)
    !
    ! one row, spectral -> physical: h(nkx) on every rank (device) -> p(mx) on rank 0 (device), with
    ! zero padding. The row is small: it is gathered on the host instead of using the global transpose.
    !
    implicit none
    complex(rp), intent(in)  :: h(nkx)
    real(rp)   , intent(out) :: p(mx)
    integer :: i
#if defined(_CUDA)
    integer :: istat
#endif
    !$acc wait(1)
    !$acc update self(h)
    call row_gather(h,rglob)
    if(myid /= 0) return
    rspec(1:nxh) = rglob(1:nxh)
    do i=nxh+1,mx/2+1
      rspec(i) = (0._rp,0._rp)
    end do
    !$acc update device(rspec)
#if defined(_CUDA)
    !$acc host_data use_device(p,rspec)
    istat = cufftExecZ2D(plan_rb,rspec,p)
    !$acc end host_data
#else
    call fftw_execute_dft_c2r(plan_rb,rspec,p)
#endif
  end subroutine row_bwd
  !
  subroutine row_fwd(p,h)
    ! one row, physical -> spectral: p(mx) on rank 0 (device) -> h(nkx) on every rank (device)
    implicit none
    real(rp)   , intent(inout) :: p(mx)
    complex(rp), intent(out)   :: h(nkx)
#if defined(_CUDA)
    integer :: istat
#endif
    if(myid == 0) then
#if defined(_CUDA)
      !$acc host_data use_device(p,rspec)
      istat = cufftExecD2Z(plan_rf,p,rspec)
      !$acc end host_data
#else
      !$acc wait(1)
      call fftw_execute_dft_r2c(plan_rf,p,rspec)
#endif
      !$acc wait(1)
      !$acc update self(rspec)
      rglob(1:nxh) = rspec(1:nxh)*norm
    end if
    call row_scatter(rglob,h)
    !$acc update device(h)
  end subroutine row_fwd
end module mod_fft
