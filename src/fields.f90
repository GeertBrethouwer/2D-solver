!
! field storage
!
module mod_fields
  use mod_types
  use mod_param , only: ny,mx,nscal,nl_form,scal_form
  use mod_decomp, only: nkx,nyA
  implicit none
  public
  ! spectral (y-lines): velocity, nonlinear terms N_x, N_y, the previous RK right-hand side of phi,
  ! work arrays; phi itself is phv in mod_ysolve
  complex(rp), allocatable, dimension(:,:)   :: u,v,nu,nv,nvo,tmp1,tmp2
  complex(rp), allocatable, dimension(:,:,:) :: s,ns,nso
  ! physical (x-lines on the dealiased grid)
  real(rp), allocatable, dimension(:,:)   :: up,vp,ap
  real(rp), allocatable, dimension(:,:,:) :: gp
contains
  subroutine alloc_fields
    implicit none
    integer :: ng
    allocate(u(nkx,ny),v(nkx,ny),nu(nkx,ny),nv(nkx,ny),nvo(nkx,ny),tmp1(nkx,ny),tmp2(nkx,ny))
    allocate(s(nkx,ny,max(nscal,1)),ns(nkx,ny,max(nscal,1)),nso(nkx,ny,max(nscal,1)))
    ! work arrays gp: the divergence and skew scalar forms need 2 (add_flux_div), else 1
    ng = 1
    if(nscal > 0 .and. trim(scal_form) /= 'advective') ng = 2
    allocate(up(mx,nyA),vp(mx,nyA),ap(mx,nyA),gp(mx,nyA,ng))
    u = 0; v = 0; nu = 0; nv = 0; nvo = 0; tmp1 = 0; tmp2 = 0
    s = 0; ns = 0; nso = 0
    up = 0; vp = 0; ap = 0; gp = 0
    !$acc enter data copyin(u,v,nu,nv,nvo,tmp1,tmp2,s,ns,nso,up,vp,ap,gp)
  end subroutine alloc_fields
end module mod_fields
