!
! three-equation melt condition at an ice interface at the wall y = 0 (docs/three_equation_melt.tex),
! heat conduction into the ice neglected. In the units of the code, at every x:
!
!   theta_b = Lambda s_b + theta_f                      liquidus
!   m*      = St kappa_T dtheta/dy|_0                   heat:  latent heat = conductive heat flux
!   kappa_S ds/dy|_0 = m* (s_b + sigma)                 salt:  diffusive flux = dilution by meltwater
!
! with m* = rho_i m/(rho_w U) the melt rate as a water velocity, kappa = 1/(Re Pr), and the absolute
! salinity S_b - S_ice = Delta S (s_b + sigma). With melt_blowing the interface is not impermeable:
! v(x,0) = m*(x), u(x,0) = 0 (Wells & Worster, J. Fluid Mech. 687, 2011, Eq. 3.5a).
!
! Implicit coupling (melt_solve). The interface values theta_b(x), s_b(x) are the Dirichlet values of
! the scalar solves of the RK substep, and the conditions must hold with the wall gradients of the NEW
! scalars. The theta-method solve is linear in the wall value: for every Fourier mode k the new wall
! gradient is g_k = g0_k + R_k w_k, with w the wall value and R_k < 0 the response of the wall gradient
! to a unit wall value (computed with the solver itself; it depends on k and on the diffusion
! coefficient c of the substep, and is cached). The coupled problem is solved by the iteration
!
!   1. solve the scalars with the current wall values w (all modes), wall gradients g;
!   2. G = g - R_ref w with a constant R_ref (per scalar); at every x solve the conditions with the
!      wall gradient G + R_ref w_new (local; a quadratic for the salinity, see below);
!   3. repeat until max|w_new - w| < melt_tol.
!
! At convergence the conditions hold exactly with the new gradients. Per mode the error is multiplied
! by at most |1 - R_k/R_ref|, which is < 1 for R_ref = (min R_k + max R_k)/2, whatever the time step.
!
! Local problem. With a = R_ref, z = s_b + sigma and G_z = G_s - a sigma (mean mode):
!
!   B z^2 + (A - kappa_S a_S) z - kappa_S G_z = 0,  A = St kappa_T (G_theta + a_T (theta_f - Lambda sigma)),
!                                                   B = St kappa_T a_T Lambda >= 0,
!
! whose roots have opposite signs (a < 0, Lambda <= 0, G_z > 0 for positive salinity in the water): the
! physical root is the non-negative one. Then s_b = z - sigma, theta_b = Lambda s_b + theta_f,
! m* = A + B z. Without salinity (melt_s = 0): theta_b = theta_f, m* = St kappa_T (G_theta + a_T theta_f)
! (Stefan condition). The products are formed in physical space (dealiased grid), on the rank that
! holds the wall row.
!
module mod_melt
  use mpi
  use mod_types
  use mod_param
  use mod_decomp , only: myid,nkx,nyA,yA_lo
  use mod_fft    , only: fwd_fft,bwd_fft
  use mod_ysolve , only: c1,ismean,mean_owner,i00,swb,vbw,scalar_solve
  use mod_compact, only: nc1
  implicit none
  private
  public :: init_melt,melt_solve,melt_means,melt_output,melt_nit,melt_res
  integer :: nsb
  complex(rp), allocatable :: mh(:,:,:)   ! (nkx,ny,3) spectral work; row 1 holds theta_b, s_b, m* after melt_solve
  real(rp)   , allocatable :: mp(:,:,:)   ! (mx,nyA,3) physical work
  complex(rp), allocatable :: mr(:,:,:)   ! (nkx,ny,nsb) right-hand sides of the scalar solves (kept for the iteration)
  complex(rp), allocatable :: mt(:,:,:)   ! (nkx,ny,nsb) work field for the wall response
  complex(rp), allocatable :: wo(:,:)     ! (nkx,2) previous iterate of the wall values
  real(rp)   , allocatable :: resp(:,:,:) ! (nkx,2,3) wall response R_k of temperature and salinity per substep
  real(rp) :: rref(2,3) = 0._rp, c_cached(3) = -1._rp
  integer  :: melt_nit = 0                ! iterations in the last substep
  real(rp) :: melt_res = 0._rp            ! last change of the wall values
contains
  subroutine init_melt
    implicit none
    nsb = max(nscal,1)
    allocate(mh(nkx,ny,3),mp(mx,nyA,3),mr(nkx,ny,nsb),mt(nkx,ny,nsb),wo(nkx,2),resp(nkx,2,3))
    mh = 0; mp = 0; mr = 0; mt = 0; wo = 0; resp = 0
    !$acc enter data copyin(mh,mp,mr,mt,wo,resp)
  end subroutine init_melt
  !
  subroutine wall_response(q,dtb)
    !
    ! R_k = wall gradient of the theta-method solution with zero right-hand side and unit wall value at
    ! y = 0 (all modes), for the temperature (1) and the salinity (2), and R_ref = (min R_k + max R_k)/2
    !
    implicit none
    integer , intent(in) :: q
    real(rp), intent(in) :: dtb
    integer :: i,r,n,j,it,is,ierr
    real(rp) :: rmin(2),rmax(2),g,rmin1,rmin2,rmax1,rmax2
    it = melt_t; is = melt_s
    !$acc parallel loop collapse(2) default(present) async(1)
    do i=1,nkx
      do j=1,2
        wo(i,j) = swb(i,merge(it,max(is,1),j == 1))      ! save the wall values
      end do
    end do
    !$acc parallel loop collapse(3) default(present) async(1)
    do n=1,nsb
      do r=1,ny
        do i=1,nkx
          mt(i,r,n) = (0._rp,0._rp)
        end do
      end do
    end do
    !$acc parallel loop default(present) async(1)
    do i=1,nkx
      swb(i,it) = (1._rp,0._rp)
      if(is > 0) swb(i,is) = (1._rp,0._rp)
    end do
    call scalar_solve(mt,nscal,dtb)
    rmin1 = huge(1._rp); rmax1 = -huge(1._rp); rmin2 = huge(1._rp); rmax2 = -huge(1._rp)
    !$acc parallel loop default(present) private(g) reduction(min:rmin1,rmin2) reduction(max:rmax1,rmax2) async(1)
    do i=1,nkx
      do j=1,2
        n = merge(it,max(is,1),j == 1)
        g = 0._rp
        do r=1,nc1
          g = g + c1(r)*real(mt(i,r,n),rp)
        end do
        resp(i,j,q) = g
        if(j == 1) then
          rmin1 = min(rmin1,g); rmax1 = max(rmax1,g)
        else
          rmin2 = min(rmin2,g); rmax2 = max(rmax2,g)
        end if
      end do
    end do
    !$acc wait(1)
    rmin = [rmin1,rmin2]; rmax = [rmax1,rmax2]
    call MPI_ALLREDUCE(MPI_IN_PLACE,rmin,2,MPI_DOUBLE_PRECISION,MPI_MIN,MPI_COMM_WORLD,ierr)
    call MPI_ALLREDUCE(MPI_IN_PLACE,rmax,2,MPI_DOUBLE_PRECISION,MPI_MAX,MPI_COMM_WORLD,ierr)
    rref(:,q) = 0.5_rp*(rmin(:)+rmax(:))
    !$acc parallel loop default(present) async(1)
    do i=1,nkx
      swb(i,it) = wo(i,1)                                 ! restore the wall values
      if(is > 0) swb(i,is) = wo(i,2)
    end do
  end subroutine wall_response
  !
  subroutine melt_solve(s,q,dtb)
    !
    ! theta-method solves of all scalars (on input s holds the right-hand sides) with the melt condition
    ! at y = 0 imposed implicitly; sets swb (scalar wall values) and vbw (wall-normal velocity)
    !
    implicit none
    complex(rp), intent(inout) :: s(nkx,ny,max(nscal,1))
    integer , intent(in) :: q
    real(rp), intent(in) :: dtb
    integer :: i,r,n,it,is,iter,ierr
    real(rp) :: aT,aS,kt,ks,st,lam,thf,sig,gth,gz,aa,bb,bq,cq,disc,z,sb,thb,mm,dw
    complex(rp) :: g1,g2
    logical :: salt,blow
    it = melt_t; is = melt_s; salt = (is > 0); blow = melt_blowing
    kt = diff(it); ks = 0._rp
    if(salt) ks = diff(is)
    st = melt_st; lam = melt_lambda; thf = melt_thf; sig = melt_sigma
    if(dtb /= c_cached(q)) then
      call wall_response(q,dtb)
      c_cached(q) = dtb
    end if
    aT = rref(1,q); aS = rref(2,q)
    !
    ! keep the right-hand sides
    !
    !$acc parallel loop collapse(3) default(present) async(1)
    do n=1,nsb
      do r=1,ny
        do i=1,nkx
          mr(i,r,n) = s(i,r,n)
        end do
      end do
    end do
    do iter=1,melt_maxit
      if(iter > 1) then
        !$acc parallel loop collapse(3) default(present) async(1)
        do n=1,nsb
          do r=1,ny
            do i=1,nkx
              s(i,r,n) = mr(i,r,n)
            end do
          end do
        end do
      end if
      call scalar_solve(s,nscal,dtb)
      !
      ! G = g - R_ref w in spectral space (row 1 of mh); w = theta_b and z = s_b + sigma
      !
      !$acc parallel loop gang vector default(present) private(g1,g2) async(1)
      do i=1,nkx
        g1 = (0._rp,0._rp); g2 = (0._rp,0._rp)
        do r=1,nc1
          g1 = g1 + c1(r)*s(i,r,it)
        end do
        g1 = g1 - aT*swb(i,it)
        wo(i,1) = swb(i,it)
        if(salt) then
          do r=1,nc1
            g2 = g2 + c1(r)*s(i,r,is)
          end do
          g2 = g2 - aS*swb(i,is)
          if(ismean(i)) g2 = g2 - aS*sig
          wo(i,2) = swb(i,is)
        end if
        do r=1,ny
          mh(i,r,1) = (0._rp,0._rp); mh(i,r,2) = (0._rp,0._rp)
        end do
        mh(i,1,1) = g1; mh(i,1,2) = g2
      end do
      call bwd_fft(mh(:,:,1),mp(:,:,1))
      call bwd_fft(mh(:,:,2),mp(:,:,2))
      !
      ! interface conditions at every x of the wall row (on the rank that holds it)
      !
      !$acc kernels default(present) async(1)
      mp(:,:,3) = 0._rp
      !$acc end kernels
      if(yA_lo == 1) then
        !$acc parallel loop default(present) async(1)
        do i=1,mx
          gth = mp(i,1,1); gz = mp(i,1,2)
          if(salt) then
            aa = st*kt*(gth + aT*(thf - lam*sig))
            bb = st*kt*aT*lam
            bq = aa - ks*aS
            cq = ks*gz
            if(bb > 0._rp) then
              disc = max(bq*bq + 4._rp*bb*cq,0._rp)
              if(bq >= 0._rp) then
                if(bq + sqrt(disc) > 0._rp) then
                  z = 2._rp*cq/(bq + sqrt(disc))
                else
                  z = 0._rp
                end if
              else
                z = (-bq + sqrt(disc))/(2._rp*bb)
              end if
            else if(bq > 0._rp) then
              z = cq/bq
            else
              z = 0._rp
            end if
            z = max(z,0._rp)
            sb  = z - sig
            thb = lam*sb + thf
            mm  = aa + bb*z
          else
            thb = thf
            sb  = 0._rp
            mm  = st*kt*(gth + aT*thf)
          end if
          mp(i,1,1) = thb; mp(i,1,2) = sb; mp(i,1,3) = mm
        end do
      end if
      call fwd_fft(mp(:,:,1),mh(:,:,1))
      call fwd_fft(mp(:,:,2),mh(:,:,2))
      call fwd_fft(mp(:,:,3),mh(:,:,3))
      !
      ! new wall values and their change
      !
      dw = 0._rp
      !$acc parallel loop default(present) reduction(max:dw) async(1)
      do i=1,nkx
        swb(i,it) = mh(i,1,1)
        dw = max(dw,abs(mh(i,1,1)-wo(i,1)))
        if(salt) then
          swb(i,is) = mh(i,1,2)
          dw = max(dw,abs(mh(i,1,2)-wo(i,2)))
        end if
      end do
      !$acc wait(1)
      call MPI_ALLREDUCE(MPI_IN_PLACE,dw,1,MPI_DOUBLE_PRECISION,MPI_MAX,MPI_COMM_WORLD,ierr)
      melt_nit = iter; melt_res = dw
      if(dw < melt_tol) exit
    end do
    !
    ! wall-normal velocity at the interface
    !
    !$acc parallel loop default(present) async(1)
    do i=1,nkx
      if(blow) then
        vbw(i) = mh(i,1,3)
      else
        vbw(i) = (0._rp,0._rp)
      end if
    end do
  end subroutine melt_solve
  !
  subroutine melt_means(mmean)
    ! x-averages of m*, theta_b and s_b (all ranks)
    implicit none
    real(rp), intent(out) :: mmean(3)
    integer :: ierr
    mmean(:) = 0._rp
    if(mean_owner) then
      !$acc wait(1)
      !$acc update self(mh(i00:i00,1:1,1:3))
      mmean(1) = real(mh(i00,1,3),rp)
      mmean(2) = real(mh(i00,1,1),rp)
      mmean(3) = real(mh(i00,1,2),rp)
    end if
    call MPI_ALLREDUCE(MPI_IN_PLACE,mmean,3,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,ierr)
  end subroutine melt_means
  !
  subroutine melt_output(istep)
    ! data_dir/melt_NNNNNNN.dat: x, m*, theta_b, s_b along the interface (values of the last RK substep)
    implicit none
    integer, intent(in) :: istep
    integer :: i,iunit,k
    character(len=7) :: cstep
    do k=1,3
      call bwd_fft(mh(:,:,k),mp(:,:,k))
    end do
    if(yA_lo == 1) then
      !$acc wait(1)
      !$acc update self(mp(:,1:1,1:3))
      write(cstep,'(i7.7)') istep
      open(newunit=iunit,file=trim(data_dir)//'/melt_'//cstep//'.dat',status='replace')
      write(iunit,'(a)') '#             x              m*         theta_b             s_b'
      do i=1,mx
        write(iunit,'(4es16.7e3)') lx*(i-1)/mx,mp(i,1,3),mp(i,1,1),mp(i,1,2)
      end do
      close(iunit)
    end if
  end subroutine melt_output
end module mod_melt
