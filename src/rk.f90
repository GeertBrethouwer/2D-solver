!
! nonlinear terms and the RK3 time step (Spalart, Moser & Rogers 1991)
!
!   (I - b_q dt L) f^{q+1} = (I + a_q dt L) f^q + dt [ gam_q N^q + zet_q N^{q-1} ]
!
! for phi = lap(v) and the mean flow U (L = nu (D2 - kx^2), always implicit). N contains the
! advection terms (pseudo-spectral, 3/2-rule dealiasing in x), the buoyancy and the mean pressure
! gradient. Scalars use the theta-method for diffusion; theta = 0 puts the diffusion into N (RK3).
!
module mod_rk
  use mpi
  use mod_types
  use mod_param
  use mod_decomp , only: nkx,nyA,yA_lo
  use mod_fields
  use mod_fft    , only: fwd_fft,bwd_fft
  use mod_timer  , only: tic,toc,tm_nl,tm_dt,tm_vsolve,tm_ssolve
  use mod_ysolve , only: deriv1,deriv1m,add_lap,scalar_solve,kxv,mean_owner,i00,mean_gradient_max, &
                         vv_rhs_terms,vv_step,vv_cfr,phv
  use mod_compact, only: kappa1,lam2max
  implicit none
  private
  public :: init_rk,rk_step,dpdx_applied
  real(rp) :: dpdx_applied = 0._rp
  real(rp), allocatable :: dyc_d(:)
contains
  subroutine init_rk
    use mod_compact, only: dyc
    implicit none
    allocate(dyc_d,source=dyc)
    !$acc enter data copyin(dyc_d)
    if(forcing == 'cpg') dpdx_applied = dpdx
  end subroutine init_rk
  !
  subroutine spec_mul(f,g)
    ! g = i kx f
    implicit none
    complex(rp), intent(in)  :: f(nkx,ny)
    complex(rp), intent(out) :: g(nkx,ny)
    integer :: i,k
    !$acc parallel loop collapse(2) default(present) async(1)
    do k=1,ny
      do i=1,nkx
        g(i,k) = cmplx(0._rp,kxv(i),rp)*f(i,k)
      end do
    end do
  end subroutine spec_mul
  !
  subroutine spec_axpy(f,g,ax,c)
    ! g = c*g + i ax kx f
    implicit none
    complex(rp), intent(in)    :: f(nkx,ny)
    complex(rp), intent(inout) :: g(nkx,ny)
    real(rp), intent(in) :: ax,c
    integer :: i,k
    !$acc parallel loop collapse(2) default(present) async(1)
    do k=1,ny
      do i=1,nkx
        g(i,k) = c*g(i,k) + cmplx(0._rp,ax*kxv(i),rp)*f(i,k)
      end do
    end do
  end subroutine spec_axpy
  !
  subroutine phys_prod(a,b,c,fac,accum)
    ! c = fac*a*b (accum = .false.) or c = c + fac*a*b
    implicit none
    real(rp), intent(in) :: a(mx,nyA),b(mx,nyA),fac
    real(rp), intent(inout) :: c(mx,nyA)
    logical, intent(in) :: accum
    integer :: i,k
    if(accum) then
      !$acc parallel loop collapse(2) default(present) async(1)
      do k=1,nyA
        do i=1,mx
          c(i,k) = c(i,k) + fac*a(i,k)*b(i,k)
        end do
      end do
    else
      !$acc parallel loop collapse(2) default(present) async(1)
      do k=1,nyA
        do i=1,mx
          c(i,k) = fac*a(i,k)*b(i,k)
        end do
      end do
    end if
  end subroutine phys_prod
  !
  subroutine advect(f,fac,metric)
    ! ap = fac*(u df/dx + v df/dy) in physical space (advective form);
    ! metric: d/dy through the uniform computational grid (deriv1m) instead of the compact D1
    implicit none
    complex(rp), intent(in)  :: f(nkx,ny)
    real(rp)   , intent(in)  :: fac
    logical    , intent(in)  :: metric
    call spec_mul(f,tmp1)
    call bwd_fft(tmp1,gp(:,:,1))
    call phys_prod(up,gp(:,:,1),ap,fac,.false.)
    if(metric) then
      call deriv1m(f,tmp1)
    else
      call deriv1(f,tmp1)
    end if
    call bwd_fft(tmp1,gp(:,:,1))
    call phys_prod(vp,gp(:,:,1),ap,fac,.true.)
  end subroutine advect
  !
  subroutine add_flux_div(f,nf,fac,metric)
    ! nf <- nf + fac*div(u f) (divergence form; up, vp hold the velocity in physical space)
    implicit none
    complex(rp), intent(in)    :: f(nkx,ny)
    complex(rp), intent(inout) :: nf(nkx,ny)
    real(rp)   , intent(in)    :: fac
    logical    , intent(in)    :: metric
    integer :: i,k
    call bwd_fft(f,gp(:,:,1))
    call phys_prod(up,gp(:,:,1),gp(:,:,2),1._rp,.false.)
    call fwd_fft(gp(:,:,2),tmp1)
    call spec_axpy(tmp1,nf,fac,1._rp)                   ! + fac i kx (u f)
    call phys_prod(vp,gp(:,:,1),gp(:,:,2),1._rp,.false.)
    call fwd_fft(gp(:,:,2),tmp1)
    if(metric) then
      call deriv1m(tmp1,tmp2)
    else
      call deriv1(tmp1,tmp2)
    end if
    !$acc parallel loop collapse(2) default(present) async(1)
    do k=1,ny
      do i=1,nkx
        nf(i,k) = nf(i,k) + fac*tmp2(i,k)               ! + fac D1 (v f)
      end do
    end do
  end subroutine add_flux_div
  !
  subroutine nonlinear(is_first,dtnew)
    implicit none
    logical , intent(in)  :: is_first
    real(rp), intent(out) :: dtnew
    integer :: n,i,k
    real(rp) :: qc,qu,fa
    logical  :: skew
    call bwd_fft(u,up)
    call bwd_fft(v,vp)
    dtnew = dt
    if(is_first .and. .not.fixed_dt) then
      call tic(tm_dt); dtnew = compute_dt(); call toc(tm_dt)
    end if
    !
    ! scalars: 'advective' -u.grad(s), 'divergence' -div(u s), 'skew' the average of both
    ! (conserves the scalar variance discretely; d/dy through the uniform computational grid)
    !
    skew = .false.
    select case(scal_form)
    case('divergence')
      fa = 0._rp
    case('skew')
      fa = 0.5_rp; skew = .true.
    case default
      fa = 1._rp
    end select
    do n=1,nscal
      if(fa > 0._rp) then
        call advect(s(:,:,n),-fa,skew)
      else
        !$acc kernels default(present) async(1)
        ap(:,:) = 0._rp
        !$acc end kernels
      end if
      qc = ssrc_const(n); qu = ssrc_u(n)/ubulk
      if(qc /= 0._rp .or. qu /= 0._rp) then
        !$acc parallel loop collapse(2) default(present) async(1)
        do k=1,nyA
          do i=1,mx
            ap(i,k) = ap(i,k) + qc + qu*up(i,k)
          end do
        end do
      end if
      call fwd_fft(ap,ns(:,:,n))
      if(fa < 1._rp) call add_flux_div(s(:,:,n),ns(:,:,n),fa-1._rp,skew)
      if(scal_theta(n) == 0._rp) call add_lap(s(:,:,n),ns(:,:,n),diff(n),tmp1)
    end do
    !
    ! momentum
    !
    if(trim(nl_form) == 'advective') then
      call advect(u,-1._rp,.false.)
      call fwd_fft(ap,nu)
      call advect(v,-1._rp,.false.)
      call fwd_fft(ap,nv)
    else
      ! omega_z = dv/dx - du/dy;  N = u x omega = (v omega_z, -u omega_z)
      call deriv1(u,tmp1)
      call spec_axpy(v,tmp1,1._rp,-1._rp)
      call bwd_fft(tmp1,gp(:,:,1))
      call phys_prod(vp,gp(:,:,1),ap, 1._rp,.false.)
      call fwd_fft(ap,nu)
      call phys_prod(up,gp(:,:,1),ap,-1._rp,.false.)
      call fwd_fft(ap,nv)
    end if
    !
    ! buoyancy (Boussinesq): - sum_n ri(n) s_n e_g, explicit (RK3) with the scalars of this stage
    !
    do n=1,nscal
      if(ri(n) /= 0._rp) call add_buoyancy(s(:,:,n),-ri(n)*gvec(:))
    end do
    if(forcing == 'cpg' .and. mean_owner) then
      !$acc parallel loop default(present) async(1)
      do k=1,ny
        nu(i00,k) = nu(i00,k) + dpdx
      end do
    end if
    !
    ! right-hand side h_v of the phi equation (in nv; mean column: N_x00)
    !
    call vv_rhs_terms(nu,nv)
  end subroutine nonlinear
  !
  subroutine add_buoyancy(f,b)
    ! (nu,nv) <- (nu,nv) + b(1:2)*f
    implicit none
    complex(rp), intent(in) :: f(nkx,ny)
    real(rp), intent(in) :: b(2)
    real(rp) :: bx,by
    integer :: i,k
    bx = b(1); by = b(2)
    !$acc parallel loop collapse(2) default(present) async(1)
    do k=1,ny
      do i=1,nkx
        nu(i,k) = nu(i,k) + bx*f(i,k)
        nv(i,k) = nv(i,k) + by*f(i,k)
      end do
    end do
  end subroutine add_buoyancy
  !
  real(rp) function compute_dt()
    !
    ! RK3 stability: |lambda dt| < sqrt(3) on the imaginary axis (advection) and < 2.51 on the
    ! real axis (explicit scalar diffusion, theta < 1/2); internal-gravity-wave limit
    !
    implicit none
    real(rp) :: cmax,kxm,dta,dtv,nmax
    integer :: i,k,n,ierr
    kxm = pi*nx/lx
    cmax = 0._rp
    !$acc parallel loop collapse(2) default(present) reduction(max:cmax) async(1)
    do k=1,nyA
      do i=1,mx
        cmax = max(cmax,abs(up(i,k))*kxm + abs(vp(i,k))*kappa1/dyc_d(yA_lo+k-1))
      end do
    end do
    !$acc wait(1)
    call MPI_ALLREDUCE(MPI_IN_PLACE,cmax,1,MPI_DOUBLE_PRECISION,MPI_MAX,MPI_COMM_WORLD,ierr)
    dta = cfl/max(cmax,1.e-12_rp)
    nmax = 0._rp
    do n=1,nscal
      nmax = max(nmax,diff(n)*max(0._rp,1._rp-2._rp*scal_theta(n)))
    end do
    if(nmax > 0._rp) then
      dtv = cfl_visc/(nmax*(kxm**2+lam2max))
    else
      dtv = huge(1._rp)
    end if
    compute_dt = min(dta,dtv,dt_max)
    do n=1,nscal
      if(ri(n) /= 0._rp) then
        nmax = sqrt(abs(ri(n))*mean_gradient_max(s(:,:,n)))
        if(nmax > 0._rp) compute_dt = min(compute_dt,cfl/nmax)
      end if
    end do
  end function compute_dt
  !
  subroutine update(f,nf,nfo,dtg,dtz)
    ! f = f + dtg*nf + dtz*nfo ; nfo = nf
    implicit none
    complex(rp), intent(inout) :: f(nkx,ny),nfo(nkx,ny)
    complex(rp), intent(in) :: nf(nkx,ny)
    real(rp), intent(in) :: dtg,dtz
    integer :: i,k
    !$acc parallel loop collapse(2) default(present) async(1)
    do k=1,ny
      do i=1,nkx
        f(i,k)   = f(i,k) + dtg*nf(i,k) + dtz*nfo(i,k)
        nfo(i,k) = nf(i,k)
      end do
    end do
  end subroutine update
  !
  subroutine rhs_theta(f,nf,nfo,dtg,dtz,cexp)
    !
    !   f <- f + dtg*nf + dtz*nfo + cexp*(D2 - kx^2) f ,   nfo <- nf
    !
    implicit none
    complex(rp), intent(inout) :: f(nkx,ny),nfo(nkx,ny)
    complex(rp), intent(in) :: nf(nkx,ny)
    real(rp), intent(in) :: dtg,dtz,cexp
    integer :: i,k
    !$acc parallel loop collapse(2) default(present) async(1)
    do k=1,ny
      do i=1,nkx
        tmp2(i,k) = f(i,k) + dtg*nf(i,k) + dtz*nfo(i,k)
        nfo(i,k)  = nf(i,k)
      end do
    end do
    if(cexp /= 0._rp) call add_lap(f,tmp2,cexp,tmp1)
    !$acc parallel loop collapse(2) default(present) async(1)
    do k=1,ny
      do i=1,nkx
        f(i,k) = tmp2(i,k)
      end do
    end do
  end subroutine rhs_theta
  !
  subroutine rk_step(time,dtcur)
    implicit none
    real(rp), intent(inout) :: time,dtcur
    integer :: q,n
    real(rp) :: dtn,dtg,dtz,dtb,c,dpe,th
    do q=1,3
      call tic(tm_nl)
      call nonlinear(q == 1,dtn)
      call toc(tm_nl)
      if(q == 1) then
        if(fixed_dt) then
          dtcur = dtn
        else
          !
          ! keep dt piecewise constant: the 2x2 influence matrices of the implicit scheme
          ! depend on dt and are recomputed only when it changes; an adaptive dt grows by
          ! at most dt_growth per step
          !
          if(dtcur > dtn) then
            dtcur = 0.95_rp*dtn
          else if(dtcur < 0.8_rp*dtn) then
            dtcur = min(0.95_rp*dtn,dt_growth*dtcur)
          end if
        end if
      end if
      dtg = rk_gam(q)*dtcur
      dtz = rk_zet(q)*dtcur
      dtb = rk_bet(q)*dtcur
      !
      ! velocity: (I - b dt L) phi = (I + a dt L) phi^q + dt (gam h + zet h_old)
      !
      call rhs_theta(phv,nv,nvo,dtg,dtz,rk_asmr(q)*dtcur*visc)
      c = rk_bsmr(q)*dtcur*visc
      call tic(tm_vsolve)
      call vv_step(u,v,q,c)
      if(forcing == 'cfr') then
        call vv_cfr(u,ubulk,dpe,dtb,c,q)
        if(q == 3) dpdx_applied = dpe
      end if
      call toc(tm_vsolve)
      !
      ! scalars: r = s + dt(gam N + zet N_old) + (1-theta) beta dt kappa Lap(s), then the theta-solve
      !
      do n=1,nscal
        th = scal_theta(n)
        if(th > 0._rp) then
          call rhs_theta(s(:,:,n),ns(:,:,n),nso(:,:,n),dtg,dtz,(1._rp-th)*dtb*diff(n))
          c = th*dtb*diff(n)
        else
          call update(s(:,:,n),ns(:,:,n),nso(:,:,n),dtg,dtz)
          c = 0._rp
        end if
        call tic(tm_ssolve)
        call scalar_solve(s(:,:,n),c,sbc_bot(n),sbc_top(n),sval_bot(n),sval_top(n))
        call toc(tm_ssolve)
      end do
    end do
    !$acc wait(1)
    time = time + dtcur
  end subroutine rk_step
end module mod_rk
