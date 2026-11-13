!
! wall-normal operations on spectral y-lines f(nkx,ny): compact derivatives, the scalar
! Helmholtz solves and the velocity-vorticity solver
!
! All column operations are done by one GPU thread per wavenumber kx that marches
! sequentially in y; consecutive threads access consecutive kx (coalesced). Only banded
! matrices (from the compact scheme) are factorised, on the fly, without pivoting; the
! dense operators D1 = A1^{-1}B1 and D2 = A2^{-1}B2 are never formed.
!
! Velocity-vorticity method in 2D (Kim, Moin & Moser 1987; Lluesma-Rodriguez, Alcantara-Avila,
! Perez-Quiles & Hoyas, Mathematics 9 (2021) 756), the 'vv' option of HiCHAN without z:
!
!   d/dt phi = h_v + nu (D2 - kx^2) phi ,   (D2 - kx^2) v = phi ,   v = D1 v = 0 at the walls
!   h_v = -kx^2 N_y - i kx D1 N_x                                      (kx /= 0)
!   u = i D1 v / kx   (continuity i kx u + D1 v = 0),   omega_z = i kx v - D1 u = -i phi/kx
!   d/dt U = N_x00 + nu D2 U ,  U = 0 at the walls                     (kx = 0)
!
! In 2D there is no wall-normal vorticity (omega_y = du/dz - dw/dx = 0): phi alone carries the
! dynamics of the fluctuations, phi = d omega_z/dx. N = nonlinear term + body forces (buoyancy,
! mean pressure gradient). The Neumann condition D1 v = 0 is imposed with two homogeneous
! solutions (phi = 1 at one wall) whose 2x2 matrix depends on kx and dt only (cached per RK
! substep).
!
module mod_ysolve
  use mod_types
  use mod_param , only: nxh,ny,lx,abort_run,vv_neumann,freeslip_bot,freeslip_top,hgt
  use mod_decomp, only: nkx,kx_lo
  use mod_compact, only: nbw,ca1 => a1,cb1 => b1,ca2 => a2,cb2 => b2,ct1w => t1w,ct1d => t1d,ct1u => t1u, &
                         ct2w => t2w,ct2d => t2d,ct2u => t2u,cc1 => c1,ccn => cn,cnc1 => nc1,cncn => ncn, &
                         cwq => wq,ckl2 => kl2,cku2 => ku2,cew1 => ew1,cewn => ewn,cew2 => ew2, &
                         cc21 => c21,cnc21 => nc21,cew2n => ew2n,cc2n => c2n,cnc2n => nc2n, &
                         cb1m => b1m,ct1mw => t1mw,ct1md => t1md,ct1mu => t1mu,crxi => rxi
  implicit none
  private
  public :: init_ysolve,deriv1,deriv1m,add_lap,add_kx2,scalar_solve,divergence_max,wall_gradient,mean_gradient_max, &
            mean_owner,i00,kxv,ismean, &
            init_vv,vv_state_from_velocity,vel_from_v,vv_rhs_terms,vv_step,vv_cfr
  real(rp), allocatable, public :: a1(:,:),b1(:,:),a2(:,:),b2(:,:)
  real(rp), allocatable :: t1w(:),t1d(:),t1u(:),t2w(:),t2d(:),t2u(:)
  real(rp), allocatable :: b1m(:,:),t1mw(:),t1md(:),t1mu(:),rxi(:)
  real(rp), allocatable, public :: c1(:),cn(:),wq(:)
  integer :: nc1,ncn,klh,kuh
  real(rp), allocatable :: kxv(:)
  logical , allocatable :: ismean(:)
  real(rp), allocatable :: uf(:,:,:)
  complex(rp), allocatable :: psi(:,:),sx(:,:)
  logical :: mean_owner
  integer :: i00 = -1
  !
  ! batched scalar solve: per-scalar theta, diffusivity, wall values and Neumann flags; work arrays
  !
  integer :: nsb
  real(rp), allocatable :: sth(:),sdf(:),sgb(:),sgt(:),skf(:)   ! skf: 1, or 0 without kx^2 (explicit x-diffusion)
  logical , allocatable :: snb(:),snt(:)
  complex(rp), allocatable :: sxb(:,:)
  real(rp), allocatable :: ufb(:,:,:)
  ! Dirichlet values of the scalars at the wall y = 0, per Fourier mode (nkx,nsb): the constant sval_bot
  ! in the mean mode, or the interface values of the melt model (mod_melt) in all modes
  complex(rp), allocatable, public :: swb(:,:)
  ! wall-normal velocity at y = 0 per Fourier mode (nkx): zero, or the melt (blowing) velocity
  complex(rp), allocatable, public :: vbw(:)
  !
  ! velocity-vorticity method
  !
  complex(rp), allocatable, public :: phv(:,:)  ! phi = lap(v) (mean column: U)
  complex(rp), allocatable :: wa(:,:),wb(:,:)
  real(rp), allocatable :: hinv_c(:,:,:),ginv_v(:,:,:),hinv_v(:,:)
  real(rp) :: c_cached_vv(3) = -1._rp
  real(rp), allocatable :: dw1(:),dwn(:)    ! wall functionals at y = 0 (dw1) and y = ly (dwn):
                                            ! dv/dy = 0 (no-slip) or d2v/dy2 = 0 (free-slip)
  integer :: ndw1,ndwn
contains
  subroutine init_ysolve
    implicit none
    integer :: i,ig
    allocate(a1,source=ca1); allocate(b1,source=cb1)
    allocate(a2,source=ca2); allocate(b2,source=cb2)
    allocate(t1w,source=ct1w); allocate(t1d,source=ct1d); allocate(t1u,source=ct1u)
    allocate(t2w,source=ct2w); allocate(t2d,source=ct2d); allocate(t2u,source=ct2u)
    allocate(c1,source=cc1); allocate(cn,source=ccn); allocate(wq,source=cwq)
    nc1 = cnc1; ncn = cncn
    klh = max(ckl2,1); kuh = max(cku2,1)
    !$acc enter data copyin(a1,b1,a2,b2,t1w,t1d,t1u,t2w,t2d,t2u,c1,cn,wq)
    allocate(b1m,source=cb1m); allocate(rxi,source=crxi)
    allocate(t1mw,source=ct1mw); allocate(t1md,source=ct1md); allocate(t1mu,source=ct1mu)
    !$acc enter data copyin(b1m,t1mw,t1md,t1mu,rxi)
    allocate(kxv(nkx),ismean(nkx))
    mean_owner = .false.
    do i=1,nkx
      ig = kx_lo+i-1
      kxv(i) = 2*pi/lx*(ig-1)
      ismean(i) = (ig == 1)
      if(ismean(i)) then
        mean_owner = .true.; i00 = i
      end if
    end do
    !$acc enter data copyin(kxv,ismean)
    allocate(uf(nkx,ny,0:nbw),psi(nkx,ny),sx(nkx,ny))
    uf = 0; psi = 0; sx = 0
    !$acc enter data copyin(uf,psi,sx)
    call init_scalar_solve
  end subroutine init_ysolve
  !
  ! ---------------------------------------------------------------------------
  ! column kernels (device routines)
  ! ---------------------------------------------------------------------------
  !
  subroutine bsolve(n1,n,i,kl,ku,ca,cb,a,b,rep1,repn,x,uf)
    !
    ! solves (ca*A + cb*B) x = rhs for column i, rhs given in x; A is tridiagonal, B banded
    ! with kl lower and ku upper diagonals; if rep1 (repn) the first (last) row is replaced
    ! by the identity row. Banded LU without pivoting; the U factor is stored in uf.
    !
    !$acc routine seq
    implicit none
    integer , intent(in) :: n1,n,i,kl,ku
    real(rp), intent(in) :: ca,cb
    real(rp), intent(in) :: a(-1:1,n),b(-nbw:nbw,n)
    logical , intent(in) :: rep1,repn
    complex(rp), intent(inout) :: x(n1,n)
    real(rp), intent(inout) :: uf(n1,n,0:nbw)
    real(rp) :: row(-nbw:nbw),f
    complex(rp) :: rr
    integer :: r,o,l,q
    do r=1,n
      do o=-nbw,nbw
        row(o) = 0._rp
      end do
      if((r == 1 .and. rep1) .or. (r == n .and. repn)) then
        row(0) = 1._rp
      else
        do o=-1,1
          row(o) = ca*a(o,r)
        end do
        do o=-kl,ku
          row(o) = row(o) + cb*b(o,r)
        end do
      end if
      rr = x(i,r)
      do l=max(1,r-kl),r-1
        o = l-r
        if(row(o) /= 0._rp) then
          f = row(o)/uf(i,l,0)
          do q=1,ku
            if(o+q <= ku) row(o+q) = row(o+q) - f*uf(i,l,q)
          end do
          rr = rr - f*x(i,l)
        end if
      end do
      do q=0,ku
        uf(i,r,q) = row(q)
      end do
      x(i,r) = rr
    end do
    x(i,n) = x(i,n)/uf(i,n,0)
    do r=n-1,1,-1
      rr = x(i,r)
      do q=1,min(ku,n-r)
        rr = rr - uf(i,r,q)*x(i,r+q)
      end do
      x(i,r) = rr/uf(i,r,0)
    end do
  end subroutine bsolve
  !
  pure complex(rp) function bmul(n1,n,i,r,ca,cb,a,b,x)
    ! row r of (ca*A + cb*B) x
    !$acc routine seq
    implicit none
    integer , intent(in) :: n1,n,i,r
    real(rp), intent(in) :: ca,cb,a(-1:1,n),b(-nbw:nbw,n)
    complex(rp), intent(in) :: x(n1,n)
    integer :: o
    complex(rp) :: s
    s = (0._rp,0._rp)
    do o=max(-nbw,1-r),min(nbw,n-r)
      if(abs(o) <= 1) then
        s = s + (ca*a(o,r)+cb*b(o,r))*x(i,r+o)
      else
        s = s + cb*b(o,r)*x(i,r+o)
      end if
    end do
    bmul = s
  end function bmul
  !
  subroutine dcol(n1,n,i,b,tw,td,tu,f,df)
    ! df = A^{-1} B f for column i (A tridiagonal, Thomas factors tw,td,tu); df /= f
    !$acc routine seq
    implicit none
    integer , intent(in) :: n1,n,i
    real(rp), intent(in) :: b(-nbw:nbw,n),tw(n),td(n),tu(n)
    complex(rp), intent(in) :: f(n1,n)
    complex(rp), intent(inout) :: df(n1,n)
    integer :: r,o
    complex(rp) :: s
    do r=1,n
      s = (0._rp,0._rp)
      do o=max(-nbw,1-r),min(nbw,n-r)
        s = s + b(o,r)*f(i,r+o)
      end do
      if(r > 1) s = s - tw(r)*df(i,r-1)
      df(i,r) = s
    end do
    df(i,n) = df(i,n)*td(n)
    do r=n-1,1,-1
      df(i,r) = (df(i,r) - tu(r)*df(i,r+1))*td(r)
    end do
  end subroutine dcol
  !
  subroutine a2rhs(n1,n,i,a2,g,x)
    ! x = A2 g~ at rows 2..N-1 (g~ = g with zero wall values), x = 0 at rows 1 and N
    !$acc routine seq
    implicit none
    integer , intent(in) :: n1,n,i
    real(rp), intent(in) :: a2(-1:1,n)
    complex(rp), intent(in) :: g(n1,n)
    complex(rp), intent(inout) :: x(n1,n)
    integer :: r,q
    complex(rp) :: s
    do r=2,n-1
      s = (0._rp,0._rp)
      do q=max(-1,2-r),min(1,n-1-r)
        s = s + a2(q,r)*g(i,r+q)
      end do
      x(i,r) = s
    end do
    x(i,1) = (0._rp,0._rp); x(i,n) = (0._rp,0._rp)
  end subroutine a2rhs
  !
  subroutine hsetup(n1,n,i,kl,ku,ca,cb,a2,b2,x,uf,h)
    !
    ! inverse of the 2x2 matrix that determines the unknown wall values tau_1, tau_N of
    ! L s in the A2-premultiplied Helmholtz problem (the boundary error of the compact scheme)
    !
    !$acc routine seq
    implicit none
    integer , intent(in) :: n1,n,i,kl,ku
    real(rp), intent(in) :: ca,cb,a2(-1:1,n),b2(-nbw:nbw,n)
    complex(rp), intent(inout) :: x(n1,n)
    real(rp), intent(inout) :: uf(n1,n,0:nbw)
    real(rp), intent(out) :: h(4)
    real(rp) :: m11,m12,m21,m22,det
    integer :: r
    do r=1,n
      x(i,r) = (0._rp,0._rp)
    end do
    x(i,2) = a2(-1,2)
    call bsolve(n1,n,i,kl,ku,ca,cb,a2,b2,.true.,.true.,x,uf)
    m11 = real(bmul(n1,n,i,1,ca,cb,a2,b2,x),rp) - a2(0,1)
    m21 = real(bmul(n1,n,i,n,ca,cb,a2,b2,x),rp)
    do r=1,n
      x(i,r) = (0._rp,0._rp)
    end do
    x(i,n-1) = a2(1,n-1)
    call bsolve(n1,n,i,kl,ku,ca,cb,a2,b2,.true.,.true.,x,uf)
    m12 = real(bmul(n1,n,i,1,ca,cb,a2,b2,x),rp)
    m22 = real(bmul(n1,n,i,n,ca,cb,a2,b2,x),rp) - a2(0,n)
    det = m11*m22 - m12*m21
    h(1) =  m22/det; h(2) = -m21/det; h(3) = -m12/det; h(4) = m11/det
  end subroutine hsetup
  !
  subroutine hsolve_ab(n1,n,i,kl,ku,ca,cb,a2,b2,h,g,x,uf,wa_,wb_)
    !
    ! exact solution of  L x = g  at interior points, x(1) = wa_, x(n) = wb_, with
    ! L = A2^{-1} (ca A2 + cb B2); the unknown wall values of L x are found with h (hsetup)
    !
    !$acc routine seq
    implicit none
    integer , intent(in) :: n1,n,i,kl,ku
    real(rp), intent(in) :: ca,cb,a2(-1:1,n),b2(-nbw:nbw,n),h(4)
    complex(rp), intent(in) :: g(n1,n),wa_,wb_
    complex(rp), intent(inout) :: x(n1,n)
    real(rp), intent(inout) :: uf(n1,n,0:nbw)
    complex(rp) :: e1,e2,t1,tn
    call a2rhs(n1,n,i,a2,g,x)
    x(i,1) = wa_; x(i,n) = wb_
    call bsolve(n1,n,i,kl,ku,ca,cb,a2,b2,.true.,.true.,x,uf)
    e1 = bmul(n1,n,i,1,ca,cb,a2,b2,x) - a2( 1,1)*g(i,2)
    e2 = bmul(n1,n,i,n,ca,cb,a2,b2,x) - a2(-1,n)*g(i,n-1)
    t1 = -(h(1)*e1 + h(3)*e2)
    tn = -(h(2)*e1 + h(4)*e2)
    call a2rhs(n1,n,i,a2,g,x)
    x(i,2)   = x(i,2)   + t1*a2(-1,2)
    x(i,n-1) = x(i,n-1) + tn*a2( 1,n-1)
    x(i,1) = wa_; x(i,n) = wb_
    call bsolve(n1,n,i,kl,ku,ca,cb,a2,b2,.true.,.true.,x,uf)
  end subroutine hsolve_ab
  !
  subroutine hsolve_mean(n1,n,i,kl,ku,ca,cb,a2,b2,h,g,x,xh,gz,uf,c1,nc1,cn,ncn,fsb,fst)
    !
    ! mean-flow solve: L x = g at interior points; at each wall x = 0 (no-slip) or D1 x = 0
    ! (free-slip: fsb at y = 0, fst at y = ly; at most one), by superposition of the solution
    ! with x = 1 at that wall; D1 at the wall = first/last row of the compact D1. xh and gz are
    ! work columns.
    !
    !$acc routine seq
    implicit none
    integer , intent(in) :: n1,n,i,kl,ku,nc1,ncn
    real(rp), intent(in) :: ca,cb,a2(-1:1,n),b2(-nbw:nbw,n),h(4),c1(n),cn(n)
    complex(rp), intent(in) :: g(n1,n)
    complex(rp), intent(inout) :: x(n1,n),xh(n1,n),gz(n1,n)
    real(rp), intent(inout) :: uf(n1,n,0:nbw)
    logical , intent(in) :: fsb,fst
    complex(rp) :: zero,one,s1,s2,al
    integer :: r
    zero = (0._rp,0._rp); one = (1._rp,0._rp)
    call hsolve_ab(n1,n,i,kl,ku,ca,cb,a2,b2,h,g,x,uf,zero,zero)
    if(fsb .or. fst) then
      do r=1,n
        gz(i,r) = zero
      end do
      s1 = zero; s2 = zero
      if(fsb) then
        call hsolve_ab(n1,n,i,kl,ku,ca,cb,a2,b2,h,gz,xh,uf,one,zero)
        do r=1,nc1
          s1 = s1 + c1(r)*x(i,r)
          s2 = s2 + c1(r)*xh(i,r)
        end do
      else
        call hsolve_ab(n1,n,i,kl,ku,ca,cb,a2,b2,h,gz,xh,uf,zero,one)
        do r=n-ncn+1,n
          s1 = s1 + cn(r)*x(i,r)
          s2 = s2 + cn(r)*xh(i,r)
        end do
      end if
      al = -s1/s2
      do r=1,n
        x(i,r) = x(i,r) + al*xh(i,r)
      end do
    end if
  end subroutine hsolve_mean
  !
  ! ---------------------------------------------------------------------------
  ! derivatives
  ! ---------------------------------------------------------------------------
  !
  subroutine deriv1(f,df)
    ! df = D1 f = A1^{-1} B1 f
    implicit none
    complex(rp), intent(in)  :: f(nkx,ny)
    complex(rp), intent(out) :: df(nkx,ny)
    integer :: i
    !$acc parallel loop gang vector default(present) async(1)
    do i=1,nkx
      call dcol(nkx,ny,i,b1,t1w,t1d,t1u,f,df)
    end do
  end subroutine deriv1
  !
  subroutine deriv1m(f,df)
    ! df = D1m f = diag(1/y_xi) A1u^{-1} B1u f: compact derivative on the uniform grid xi, mapped to y
    implicit none
    complex(rp), intent(in)  :: f(nkx,ny)
    complex(rp), intent(out) :: df(nkx,ny)
    integer :: i,r
    !$acc parallel loop gang vector default(present) async(1)
    do i=1,nkx
      call dcol(nkx,ny,i,b1m,t1mw,t1md,t1mu,f,df)
      do r=1,ny
        df(i,r) = rxi(r)*df(i,r)
      end do
    end do
  end subroutine deriv1m
  !
  subroutine add_lap(f,nf,coef,tmp,kfac)
    ! nf <- nf + coef*(D2 - kfac kx^2) f   (kfac = 1: Laplacian; 0: y-part only)
    implicit none
    complex(rp), intent(in) :: f(nkx,ny)
    complex(rp), intent(inout) :: nf(nkx,ny)
    real(rp), intent(in) :: coef,kfac
    complex(rp) :: tmp(nkx,ny)
    integer :: i,r
    real(rp) :: k2
    !$acc parallel loop gang vector default(present) private(k2) async(1)
    do i=1,nkx
      k2 = kfac*kxv(i)**2
      call dcol(nkx,ny,i,b2,t2w,t2d,t2u,f,tmp)
      do r=1,ny
        nf(i,r) = nf(i,r) + coef*(tmp(i,r) - k2*f(i,r))
      end do
    end do
  end subroutine add_lap
  !
  subroutine add_kx2(f,nf,coef)
    ! nf <- nf - coef*kx^2 f   (x-diffusion treated explicitly)
    implicit none
    complex(rp), intent(in) :: f(nkx,ny)
    complex(rp), intent(inout) :: nf(nkx,ny)
    real(rp), intent(in) :: coef
    integer :: i,r
    !$acc parallel loop collapse(2) default(present) async(1)
    do r=1,ny
      do i=1,nkx
        nf(i,r) = nf(i,r) - coef*kxv(i)**2*f(i,r)
      end do
    end do
  end subroutine add_kx2
  !
  ! ---------------------------------------------------------------------------
  ! scalars
  ! ---------------------------------------------------------------------------
  !
  subroutine init_scalar_solve
    !
    ! per-scalar parameters of the batched scalar solve, constant during the run (on the device)
    !
    use mod_param, only: nscal,scal_theta,diff,sbc_bot,sbc_top,sval_bot,sval_top,sxexp
    implicit none
    integer :: n
    nsb = max(nscal,1)
    allocate(sth(nsb),sdf(nsb),sgb(nsb),sgt(nsb),snb(nsb),snt(nsb),skf(nsb))
    sth = 0; sdf = 0; sgb = 0; sgt = 0; snb = .false.; snt = .false.; skf = 1
    do n=1,nscal
      sth(n) = scal_theta(n); sdf(n) = diff(n)
      if(sxexp(n)) skf(n) = 0._rp
      sgb(n) = sval_bot(n)  ; sgt(n) = sval_top(n)
      snb(n) = (sbc_bot(n) == 'N'); snt(n) = (sbc_top(n) == 'N')
    end do
    ! work arrays: one column per (kx, scalar) pair, column index ic = i + nkx*(n-1)
    allocate(sxb(nkx*nsb,ny),ufb(nkx*nsb,ny,0:nbw))
    sxb = 0; ufb = 0
    allocate(swb(nkx,nsb))
    swb = 0
    if(mean_owner) then
      do n=1,nscal
        swb(i00,n) = sval_bot(n)
      end do
    end if
    !$acc enter data copyin(sth,sdf,sgb,sgt,snb,snt,skf,sxb,ufb,swb)
  end subroutine init_scalar_solve
  !
  subroutine scalar_solve(s,ns_,dtb,topfac)
    !
    ! theta-method for all scalars at once:  [ (1 + c kx^2) I - c D2 ] s_n = r_n  at interior points,
    ! c = theta_n*dtb*kappa_n, with Dirichlet ('D', s = g) or Neumann ('N', D1 s = g) conditions at
    ! the walls (g only for the mean mode). On input s holds r. Premultiplying by A2 gives the
    ! banded matrix H = (1+ck^2) A2 - c B2; the unknown wall values tau_1, tau_N of L s enter rows
    ! 2 and N-1, and are determined together with the wall values of s by a small influence-matrix
    ! system (as in HiCHAN).
    !
    ! The column solves are sequential in y and latency bound on GPUs, with only nkx columns per
    ! scalar. All scalars are therefore solved in ONE kernel over the (kx, scalar) pairs: nscal*nkx
    ! threads, in about the time of a single scalar. Threads of one warp belong to the same scalar
    ! (n is the outer loop), so the Dirichlet/Neumann branches do not diverge within a warp unless
    ! nkx is not a multiple of 32.
    !
    implicit none
    integer , intent(in) :: ns_
    complex(rp), intent(inout) :: s(nkx,ny,ns_)
    real(rp), intent(in) :: dtb
    real(rp), intent(in) :: topfac       ! 1; 0 for homogeneous wall conditions at y = ly (unit responses)
    integer :: i,n,ic,r,m,q,p,l,nh,ncol
    real(rp) :: k2,c,ca,cb,gb,gt
    complex(rp) :: e(4,5),zz(4),f,t,ra1,ran,zero
    logical :: isnb,isnt,isd
    zero = (0._rp,0._rp)
    ncol = nkx*nsb
    !$acc parallel loop gang vector collapse(2) default(present) private(e,zz) async(1)
    do n=1,ns_
      do i=1,nkx
        ic = i + nkx*(n-1)
        c  = sth(n)*dtb*sdf(n)
        gb = sgb(n); gt = topfac*sgt(n)
        isnb = snb(n); isnt = snt(n)
        isd  = .not.(isnb.or.isnt)
        nh = 3
        if(.not.isd) nh = 5
        k2 = skf(n)*kxv(i)**2                 ! 0 for scalars with explicit x-diffusion
        ca = 1._rp + c*k2
        cb = -c
        ra1 = a2(1,1)*s(i,2,n)
        ran = a2(-1,ny)*s(i,ny-1,n)
        !
        ! m = 1: particular; 2: tau_1; 3: tau_N; 4: s(-1) = 1; 5: s(+1) = 1
        !
        do m=1,nh
          if(m == 1) then
            do r=2,ny-1
              sxb(ic,r) = zero
              do q=max(-1,2-r),min(1,ny-1-r)
                sxb(ic,r) = sxb(ic,r) + a2(q,r)*s(i,r+q,n)
              end do
            end do
            sxb(ic,1) = zero; sxb(ic,ny) = zero
            if(.not.isnb) sxb(ic,1) = swb(i,n)
            if(ismean(i) .and. .not.isnt) sxb(ic,ny) = gt
          else
            do r=1,ny
              sxb(ic,r) = zero
            end do
            if(m == 2) sxb(ic,2)    = a2(-1,2)
            if(m == 3) sxb(ic,ny-1) = a2( 1,ny-1)
            if(m == 4) sxb(ic,1)    = 1._rp
            if(m == 5) sxb(ic,ny)   = 1._rp
          end if
          call bsolve(ncol,ny,ic,klh,kuh,ca,cb,a2,b2,.true.,.true.,sxb,ufb)
          ! conditions: rows 1 and N of H s = A2 g, and the boundary conditions
          e(1,m) = bmul(ncol,ny,ic,1 ,ca,cb,a2,b2,sxb)
          e(2,m) = bmul(ncol,ny,ic,ny,ca,cb,a2,b2,sxb)
          if(m == 1) then
            e(1,m) = e(1,m) - ra1
            e(2,m) = e(2,m) - ran
          end if
          if(m == 2) e(1,m) = e(1,m) - a2(0,1)
          if(m == 3) e(2,m) = e(2,m) - a2(0,ny)
          if(isnb) then
            f = zero
            do r=1,nc1
              f = f + c1(r)*sxb(ic,r)
            end do
            e(3,m) = f
            if(m == 1 .and. ismean(i)) e(3,m) = e(3,m) - gb
          else
            e(3,m) = zero
            if(m == 4) e(3,m) = (1._rp,0._rp)
          end if
          if(isnt) then
            f = zero
            do r=ny-ncn+1,ny
              f = f + cn(r)*sxb(ic,r)
            end do
            e(4,m) = f
            if(m == 1 .and. ismean(i)) e(4,m) = e(4,m) - gt
          else
            e(4,m) = zero
            if(m == 5) e(4,m) = (1._rp,0._rp)
          end if
        end do
        !
        ! solve sum_m e(:,m+1) zz(m) = -e(:,1)  (Gaussian elimination, partial pivoting)
        !
        if(isd) then
          ! Dirichlet: only tau_1 and tau_N are unknown (2x2)
          f = e(1,2)*e(2,3) - e(1,3)*e(2,2)
          zz(1) = (-e(1,1)*e(2,3) + e(1,3)*e(2,1))/f
          zz(2) = (-e(2,1)*e(1,2) + e(2,2)*e(1,1))/f
          zz(3) = zero; zz(4) = zero
        else
          do q=1,4
            e(q,1) = -e(q,1)
          end do
          do q=1,4
            p = q
            do l=q+1,4
              if(abs(e(l,q+1)) > abs(e(p,q+1))) p = l
            end do
            if(p /= q) then
              do l=1,5
                t = e(q,l); e(q,l) = e(p,l); e(p,l) = t
              end do
            end if
            do p=q+1,4
              f = e(p,q+1)/e(q,q+1)
              do l=1,5
                e(p,l) = e(p,l) - f*e(q,l)
              end do
            end do
          end do
          do q=4,1,-1
            f = e(q,1)
            do l=q+1,4
              f = f - e(q,l+1)*zz(l)
            end do
            zz(q) = f/e(q,q+1)
          end do
        end if
        !
        ! final solve with the combined right-hand side
        !
        do r=2,ny-1
          sxb(ic,r) = zero
          do q=max(-1,2-r),min(1,ny-1-r)
            sxb(ic,r) = sxb(ic,r) + a2(q,r)*s(i,r+q,n)
          end do
        end do
        sxb(ic,2)    = sxb(ic,2)    + zz(1)*a2(-1,2)
        sxb(ic,ny-1) = sxb(ic,ny-1) + zz(2)*a2( 1,ny-1)
        sxb(ic,1)  = zz(3)
        sxb(ic,ny) = zz(4)
        if(.not.isnb) sxb(ic,1) = swb(i,n)
        if(ismean(i) .and. .not.isnt) sxb(ic,ny) = gt
        call bsolve(ncol,ny,ic,klh,kuh,ca,cb,a2,b2,.true.,.true.,sxb,ufb)
        do r=1,ny
          s(i,r,n) = sxb(ic,r)
        end do
      end do
    end do
  end subroutine scalar_solve
  !
  ! ---------------------------------------------------------------------------
  ! diagnostics
  ! ---------------------------------------------------------------------------
  !
  real(rp) function divergence_max(u,v,tmp)
    use mpi
    implicit none
    complex(rp), intent(in) :: u(nkx,ny),v(nkx,ny)
    complex(rp) :: tmp(nkx,ny)
    real(rp) :: dm
    integer :: i,r,ierr
    call deriv1(v,tmp)
    dm = 0._rp
    !$acc parallel loop collapse(2) default(present) reduction(max:dm) async(1)
    do r=1,ny
      do i=1,nkx
        dm = max(dm,abs(cmplx(0._rp,kxv(i),rp)*u(i,r)+tmp(i,r)))
      end do
    end do
    !$acc wait(1)
    call MPI_ALLREDUCE(MPI_IN_PLACE,dm,1,MPI_DOUBLE_PRECISION,MPI_MAX,MPI_COMM_WORLD,ierr)
    divergence_max = dm
  end function divergence_max
  !
  subroutine wall_gradient(f,gb,gt)
    ! dF/dy at both walls of the mean mode (valid on the mean owner only)
    implicit none
    complex(rp), intent(in) :: f(nkx,ny)
    real(rp), intent(out) :: gb,gt
    integer :: r
    real(rp) :: sb,st
    gb = 0._rp; gt = 0._rp
    if(.not.mean_owner) return
    sb = 0._rp; st = 0._rp
    !$acc parallel loop default(present) reduction(+:sb,st) async(1)
    do r=1,ny
      if(r <= nc1) sb = sb + c1(r)*real(f(i00,r),rp)
      if(r >= ny-ncn+1) st = st + cn(r)*real(f(i00,r),rp)
    end do
    !$acc wait(1)
    gb = sb; gt = st
  end subroutine wall_gradient
  !
  real(rp) function mean_gradient_max(f)
    ! max_y |dF/dy| of the mean (kx = 0) profile, on all ranks (for the buoyancy time-step limit)
    use mpi
    implicit none
    complex(rp), intent(in) :: f(nkx,ny)
    real(rp) :: gm
    integer :: r,ierr
    gm = 0._rp
    if(mean_owner) then
      !$acc serial default(present) async(1)
      call dcol(nkx,ny,i00,b1,t1w,t1d,t1u,f,psi)
      !$acc end serial
      !$acc parallel loop default(present) reduction(max:gm) async(1)
      do r=1,ny
        gm = max(gm,abs(real(psi(i00,r),rp)))
      end do
      !$acc wait(1)
    end if
    call MPI_ALLREDUCE(MPI_IN_PLACE,gm,1,MPI_DOUBLE_PRECISION,MPI_MAX,MPI_COMM_WORLD,ierr)
    mean_gradient_max = gm
  end function mean_gradient_max
  !
  ! ===========================================================================
  ! velocity-vorticity method
  ! ===========================================================================
  !
  subroutine init_vv
    implicit none
    integer :: i,l
    real(rp) :: h(4)
    allocate(phv(nkx,ny),wa(nkx,ny),wb(nkx,ny),vbw(nkx))
    vbw = 0
    !$acc enter data copyin(vbw)
    allocate(hinv_c(4,nkx,3),ginv_v(4,nkx,3),hinv_v(4,nkx))
    phv = 0; wa = 0; wb = 0; hinv_c = 0; ginv_v = 0; hinv_v = 0
    !$acc enter data copyin(phv,wa,wb,hinv_c,ginv_v,hinv_v)
    c_cached_vv(:) = -1._rp
    allocate(dw1(ny),dwn(ny))
    dw1(:) = 0._rp; dwn(:) = 0._rp
    if(vv_neumann == 'explicit') then
      dw1(1:6) = cew1(:); dwn(ny-5:ny) = cewn(:)
      ndw1 = 6; ndwn = 6
    else
      dw1(:) = c1(:); dwn(:) = cn(:)
      ndw1 = nc1; ndwn = ncn
    end if
    !
    ! free-slip wall: du/dy = i D2 v/kx = 0, i.e. d2v/dy2 = 0 at the wall (with v = 0), with the
    ! 7-point one-sided derivative ('explicit') or the first/last row of the compact D2 ('compact')
    !
    if(freeslip_bot) then
      dw1(:) = 0._rp
      if(vv_neumann == 'explicit') then
        dw1(1:7) = cew2(:); ndw1 = 7
      else
        dw1(:) = cc21(:); ndw1 = cnc21
      end if
    end if
    if(freeslip_top) then
      dwn(:) = 0._rp
      if(vv_neumann == 'explicit') then
        dwn(ny-6:ny) = cew2n(:); ndwn = 7
      else
        dwn(:) = cc2n(:); ndwn = cnc2n
      end if
    end if
    !$acc enter data copyin(dw1,dwn)
    !
    ! wall corrections of the (kx-only) operator  kx^2 - D2  used for (D2 - kx^2) v = phi
    !
    !$acc parallel loop gang vector default(present) private(h) async(1)
    do i=1,nkx
      if(.not.ismean(i)) then
        call hsetup(nkx,ny,i,klh,kuh,kxv(i)**2,-1._rp,a2,b2,sx,uf,h)
        do l=1,4
          hinv_v(l,i) = h(l)
        end do
      end if
    end do
    !$acc wait(1)
  end subroutine init_vv
  !
  subroutine vel_from_v(u,v)
    !
    ! u = i D1 v/kx for kx /= 0 (continuity), u = 0 at the walls; the mean column of u is kept
    !
    implicit none
    complex(rp), intent(inout) :: u(nkx,ny)
    complex(rp), intent(in)    :: v(nkx,ny)
    integer :: i,r
    real(rp) :: kx
    logical :: fsb,fst
    fsb = freeslip_bot; fst = freeslip_top
    !$acc parallel loop gang vector default(present) private(kx) async(1)
    do i=1,nkx
      if(.not.ismean(i)) then
        kx = kxv(i)
        call dcol(nkx,ny,i,b1,t1w,t1d,t1u,v,wa)
        do r=1,ny
          u(i,r) = cmplx(0._rp,1._rp/kx,rp)*wa(i,r)
        end do
        if(.not.fsb) u(i,1)  = (0._rp,0._rp)
        if(.not.fst) u(i,ny) = (0._rp,0._rp)
      end if
    end do
  end subroutine vel_from_v
  !
  subroutine vv_state_from_velocity(u,v)
    ! phi = (D2 - kx^2) v (mean column: U)
    implicit none
    complex(rp), intent(in) :: u(nkx,ny),v(nkx,ny)
    integer :: i,r
    real(rp) :: k2
    !$acc parallel loop gang vector default(present) private(k2) async(1)
    do i=1,nkx
      if(ismean(i)) then
        do r=1,ny
          phv(i,r) = u(i,r)
        end do
      else
        k2 = kxv(i)**2
        call dcol(nkx,ny,i,b2,t2w,t2d,t2u,v,phv)
        do r=1,ny
          phv(i,r) = phv(i,r) - k2*v(i,r)
        end do
      end if
    end do
    !$acc wait(1)
  end subroutine vv_state_from_velocity
  !
  subroutine vv_rhs_terms(nx_,ny_)
    !
    ! (N_x,N_y) -> h_v = -kx^2 N_y - D1(i kx N_x) in ny_; mean column: N_x00 -> ny_
    !
    implicit none
    complex(rp), intent(inout) :: nx_(nkx,ny),ny_(nkx,ny)
    integer :: i,r
    real(rp) :: kx,k2
    !$acc parallel loop gang vector default(present) private(kx,k2) async(1)
    do i=1,nkx
      if(ismean(i)) then
        do r=1,ny
          ny_(i,r) = nx_(i,r)
        end do
      else
        kx = kxv(i); k2 = kx**2
        do r=1,ny
          wb(i,r) = cmplx(0._rp,kx,rp)*nx_(i,r)
        end do
        call dcol(nkx,ny,i,b1,t1w,t1d,t1u,wb,wa)
        do r=1,ny
          ny_(i,r) = -k2*ny_(i,r) - wa(i,r)
        end do
      end if
    end do
  end subroutine vv_rhs_terms
  !
  subroutine vv_matrices(q,c)
    !
    ! for substep q, c = b_q dt nu: wall corrections of (1 + c kx^2) - c D2 and the inverse 2x2
    ! matrix [D1 v_a(-1) D1 v_b(-1); D1 v_a(1) D1 v_b(1)] of the homogeneous solutions phi_a
    ! (1 at the bottom wall), phi_b (1 at the top wall), v_a,b from (D2 - kx^2) v = phi, v = 0 at the walls
    !
    implicit none
    integer , intent(in) :: q
    real(rp), intent(in) :: c
    integer :: i,r,l,m
    real(rp) :: k2,ca,cb,h(4),hv(4),mm(2,2),det,s1,s2
    complex(rp) :: one,zero,wa_,wb_
    one = (1._rp,0._rp); zero = (0._rp,0._rp)
    !$acc parallel loop gang vector default(present) private(h,hv,mm) async(1)
    do i=1,nkx
      k2 = kxv(i)**2
      ca = 1._rp + c*k2
      cb = -c
      call hsetup(nkx,ny,i,klh,kuh,ca,cb,a2,b2,sx,uf,h)
      do l=1,4
        hinv_c(l,i,q) = h(l)
      end do
      if(.not.ismean(i)) then
        do l=1,4
          hv(l) = hinv_v(l,i)
        end do
        do m=1,2
          do r=1,ny
            wb(i,r) = zero
          end do
          wa_ = zero; wb_ = zero
          if(m == 1) wa_ = one
          if(m == 2) wb_ = one
          call hsolve_ab(nkx,ny,i,klh,kuh,ca,cb,a2,b2,h,wb,sx,uf,wa_,wb_)
          do r=1,ny
            psi(i,r) = -sx(i,r)
          end do
          call hsolve_ab(nkx,ny,i,klh,kuh,k2,-1._rp,a2,b2,hv,psi,wa,uf,zero,zero)
          s1 = 0._rp; s2 = 0._rp
          do r=1,ndw1
            s1 = s1 + dw1(r)*real(wa(i,r),rp)
          end do
          do r=ny-ndwn+1,ny
            s2 = s2 + dwn(r)*real(wa(i,r),rp)
          end do
          mm(1,m) = s1; mm(2,m) = s2
        end do
        det = mm(1,1)*mm(2,2) - mm(1,2)*mm(2,1)
        ginv_v(1,i,q) =  mm(2,2)/det; ginv_v(2,i,q) = -mm(2,1)/det
        ginv_v(3,i,q) = -mm(1,2)/det; ginv_v(4,i,q) =  mm(1,1)/det
      end if
    end do
    !$acc wait(1)
  end subroutine vv_matrices
  !
  subroutine vv_step(u,v,q,c)
    !
    ! on input phv holds the right-hand side (I + a dt L) phi^q + dt (gam h + zet h_old);
    ! on output phv, u and v at the new substep
    !
    implicit none
    complex(rp), intent(inout), dimension(nkx,ny) :: u,v
    integer , intent(in) :: q
    real(rp), intent(in) :: c
    integer :: i,r,l
    real(rp) :: kx,k2,ca,cb,h(4),hv(4)
    complex(rp) :: zero,d1,dn,cp,cm
    logical :: fsb,fst
    zero = (0._rp,0._rp)
    fsb = freeslip_bot; fst = freeslip_top
    if(c /= c_cached_vv(q)) then
      call vv_matrices(q,c)
      c_cached_vv(q) = c
    end if
    !$acc parallel loop gang vector default(present) private(h,hv) async(1)
    do i=1,nkx
      kx = kxv(i); k2 = kx**2
      ca = 1._rp + c*k2
      cb = -c
      do l=1,4
        h(l) = hinv_c(l,i,q)
      end do
      if(ismean(i)) then
        !
        ! mean flow U (kx = 0): U = 0 (no-slip) or dU/dy = 0 (free-slip) at each wall
        !
        call hsolve_mean(nkx,ny,i,klh,kuh,ca,cb,a2,b2,h,phv,sx,wa,wb,uf,c1,nc1,cn,ncn,fsb,fst)
        do r=1,ny
          phv(i,r) = sx(i,r)
          u(i,r) = sx(i,r); v(i,r) = vbw(i)    ! mean v: zero, or the mean melt velocity
        end do
      else
        do l=1,4
          hv(l) = hinv_v(l,i)
        end do
        !
        ! particular solution: phi_p = 0 at the walls, v_p = 0 at the walls
        !
        call hsolve_ab(nkx,ny,i,klh,kuh,ca,cb,a2,b2,h,phv,sx,uf,zero,zero)
        do r=1,ny
          psi(i,r) = -sx(i,r)
        end do
        call hsolve_ab(nkx,ny,i,klh,kuh,k2,-1._rp,a2,b2,hv,psi,wa,uf,vbw(i),zero)
        d1 = zero; dn = zero
        do r=1,ndw1
          d1 = d1 + dw1(r)*wa(i,r)
        end do
        do r=ny-ndwn+1,ny
          dn = dn + dwn(r)*wa(i,r)
        end do
        ! wall values of phi such that D1 v = 0 at both walls (at a free-slip wall: D2 v = 0)
        cp = -(ginv_v(1,i,q)*d1 + ginv_v(3,i,q)*dn)
        cm = -(ginv_v(2,i,q)*d1 + ginv_v(4,i,q)*dn)
        call hsolve_ab(nkx,ny,i,klh,kuh,ca,cb,a2,b2,h,phv,sx,uf,cp,cm)
        do r=1,ny
          psi(i,r) = -sx(i,r)
          phv(i,r) = sx(i,r)
        end do
        call hsolve_ab(nkx,ny,i,klh,kuh,k2,-1._rp,a2,b2,hv,psi,v,uf,vbw(i),zero)
        !
        ! u from continuity: i kx u + D1 v = 0
        !
        call dcol(nkx,ny,i,b1,t1w,t1d,t1u,v,wa)
        do r=1,ny
          u(i,r) = cmplx(0._rp,1._rp/kx,rp)*wa(i,r)
        end do
        if(.not.fsb) u(i,1)  = zero
        if(.not.fst) u(i,ny) = zero
      end if
    end do
  end subroutine vv_step
  !
  subroutine vv_cfr(u,ub_target,dpdx_eff,coef,c,q)
    !
    ! constant flow rate: U <- U + alpha h with (1 - c D2) h = 1 and the wall conditions of U
    !
    implicit none
    complex(rp), intent(inout) :: u(nkx,ny)
    real(rp), intent(in) :: ub_target,coef,c
    integer , intent(in) :: q
    real(rp), intent(out) :: dpdx_eff
    real(rp) :: ub,hb,alpha,h(4),wh
    complex(rp) :: zero
    integer :: r,l
    logical :: fsb,fst
    dpdx_eff = 0._rp
    if(.not.mean_owner) return
    zero = (0._rp,0._rp)
    fsb = freeslip_bot; fst = freeslip_top
    !$acc serial default(present) private(h) async(1)
    do l=1,4
      h(l) = hinv_c(l,i00,q)
    end do
    do r=1,ny
      wb(i00,r) = (1._rp,0._rp)
    end do
    call hsolve_mean(nkx,ny,i00,klh,kuh,1._rp,-c,a2,b2,h,wb,sx,wa,psi,uf,c1,nc1,cn,ncn,fsb,fst)
    !$acc end serial
    ub = 0._rp; hb = 0._rp
    wh = 1._rp/hgt
    !$acc parallel loop default(present) reduction(+:ub,hb) async(1)
    do r=1,ny
      ub = ub + wh*wq(r)*real(u(i00,r),rp)
      hb = hb + wh*wq(r)*real(sx(i00,r),rp)
    end do
    !$acc wait(1)
    alpha = (ub_target - ub)/hb
    !$acc parallel loop default(present) async(1)
    do r=1,ny
      u(i00,r)   = u(i00,r)   + alpha*sx(i00,r)
      phv(i00,r) = phv(i00,r) + alpha*sx(i00,r)
    end do
    dpdx_eff = alpha/coef
  end subroutine vv_cfr
end module mod_ysolve
