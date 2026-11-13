!
! input parameters (namelist file input.nml) and derived quantities
!
module mod_param
  use mpi
  use mod_types
  implicit none
  public
  integer, parameter :: nscal_max = 16
  !
  ! &grid
  !
  integer  :: nx = 64, ny = 65              ! Fourier modes in x (even), points in y (walls included)
  real(rp) :: lx = 2*pi                     ! box length
  real(rp) :: ly = 2._rp                    ! box height: walls at y = 0 and y = ly
  character(len=16)  :: grid_type = 'auto'  ! 'auto': 'tanh_top' with a free-slip bottom wall, else 'tanh';
                                            ! 'tanh' (refined at both walls), 'tanh_top' (refined at the top
                                            ! wall only), 'uniform' or 'file'
  real(rp) :: grid_alpha = 2._rp            ! stretching parameter alpha_s (tanh grids)
  character(len=256) :: grid_file = 'grid.in'
  integer  :: fd_order = 6                  ! 4 -> (3-4-3), 6 -> (3-5-6-5-3) compact schemes
  logical  :: dealias = .true.              ! 3/2 rule in x
  !
  ! &physics
  !
  real(rp) :: re = 2800._rp                 ! Reynolds number U_ref*h/nu
  character(len=3)  :: forcing = 'cfr'      ! 'cfr' (constant flow rate) or 'cpg' (constant pressure gradient)
  real(rp) :: ubulk = 1._rp                 ! bulk velocity for 'cfr'
  real(rp) :: dpdx  = 0._rp                 ! -dp/dx for 'cpg'
  character(len=8) :: vv_neumann = 'explicit' ! wall derivative in dv/dy = 0: 'explicit' (6-point one-sided)
                                            ! or 'compact' (row of D1: divergence exactly zero at the walls)
  character(len=16) :: nl_form = 'rotational' ! 'rotational' u x omega (5 FFTs) or 'advective' (8 FFTs)
  character(len=8) :: bc_bot = 'noslip'     ! velocity at the wall y = 0  : 'noslip' (u = v = 0) or
  character(len=8) :: bc_top = 'noslip'     ! velocity at the wall y = ly : 'freeslip' (v = 0, du/dy = 0)
                                            ! ('bot'/'top' = y = 0/ly, whatever the direction of gravity;
                                            ! at most one free-slip wall)
  logical  :: freeslip_bot = .false.        ! derived: bc_bot == 'freeslip'
  logical  :: freeslip_top = .false.        ! derived: bc_top == 'freeslip'
  !
  ! &time
  !
  integer  :: nstep = 1000
  real(rp) :: tmax  = huge(1._rp)
  real(rp) :: wall_time_max = huge(1._rp)   ! in seconds; the run stops (with a restart file) before this
  real(rp) :: dt    = 1.e-5_rp              ! fixed time step, or the first step of an adaptive run
  logical  :: fixed_dt = .false.
  real(rp) :: dt_max = 1._rp                ! upper bound of the adaptive time step
  real(rp) :: dt_growth = 1.1_rp            ! adaptive dt may grow by at most this factor per step
  real(rp) :: cfl = 1.0_rp, cfl_visc = 1.5_rp ! RK3: |lambda dt| < sqrt(3) (imag.) and 2.51 (real)
  !
  ! &init
  !
  character(len=16) :: init_type = 'laminar' ! 'laminar', 'noise', 'mode', 'restart'
  real(rp) :: init_amp = 0.1_rp
  integer  :: init_seed = 1
  character(len=256) :: mode_file = 'mode.in'
  integer  :: mode_kx = 1
  character(len=256) :: restart_file = 'restart'
  logical  :: reset_time = .false.
  !
  ! &output
  !
  character(len=256) :: data_dir = 'data'
  integer  :: iout_log = 10, iout_stats = 10, iout_save_stats = 1000, iout_restart = 5000, iout_field = 0
  real(rp) :: tstats_start = 0._rp
  integer  :: mode_log_kx = -1              ! log the energy of one Fourier mode (e.g. for TS waves)
  logical  :: timing = .false.              ! timers, report at the end of the run
  !
  ! &scalars (two active scalars by default: e.g. temperature and salinity)
  !
  integer  :: nscal = 2
  real(rp) :: pr(nscal_max) = 1._rp              ! Prandtl/Schmidt numbers
  real(rp) :: scal_theta(nscal_max) = 0.5_rp     ! theta of the theta-method for diffusion (0 -> explicit RK3)
  character(len=1) :: sbc_bot(nscal_max) = 'D', sbc_top(nscal_max) = 'D' ! 'D' Dirichlet or 'N' Neumann
  real(rp) :: sval_bot(nscal_max) = 1._rp, sval_top(nscal_max) = -1._rp ! wall value or wall-normal gradient
  real(rp) :: ssrc_const(nscal_max) = 0._rp      ! volumetric source  Q_c
  real(rp) :: ssrc_u(nscal_max) = 0._rp          ! source  Q_u*u/ubulk
  character(len=16) :: sinit(nscal_max) = 'linear' ! 'linear', 'zero', 'one' (uniform 1), 'noise'
  character(len=12) :: scal_form = 'advective'   ! 'advective', 'divergence' or 'skew'
  !
  ! &buoyancy (Boussinesq): momentum source  - sum_n ri(n) * s_n * e_g
  !
  real(rp) :: gravity(2) = [0._rp,-1._rp]        ! direction of gravity (x,y), normalised internally
  real(rp) :: ri(nscal_max) = 0._rp              ! Richardson number of scalar n (0 -> passive scalar)
  !
  ! &melt: three-equation melt condition at an ice interface at the wall y = 0 (docs/three_equation_melt.tex),
  ! heat conduction into the ice neglected:
  !   theta_b = Lambda s_b + theta_f,  m* = St kappa_T dtheta/dy,  kappa_S ds/dy = m* (s_b + sigma)
  ! m* = rho_i m/(rho_w U) is the melt rate as a water velocity; with melt_blowing, v(y=0) = m*
  !
  logical  :: melt = .false.
  integer  :: melt_t = 1                    ! scalar that is the temperature theta
  integer  :: melt_s = 2                    ! scalar that is the salinity s (0: no salinity -> Stefan condition)
  real(rp) :: melt_st = 0._rp               ! Stefan number St = c_w Delta T/L_f
  real(rp) :: melt_lambda = 0._rp           ! liquidus slope Lambda = lambda_1 Delta S/Delta T (<= 0)
  real(rp) :: melt_thf = 0._rp              ! liquidus offset theta_f = (lambda_1 S_ref + lambda_2 - T_ref)/Delta T
  real(rp) :: melt_sigma = 0._rp            ! sigma = (S_ref - S_ice)/Delta S
  logical  :: melt_blowing = .true.         ! v = m* at the interface (Wells & Worster 2011, Eq. 3.5a)
  character(len=8) :: melt_xdiff = 'explicit' ! x-diffusion of the temperature and salinity:
                                            ! 'explicit' (RK3; the y-solve is then the same for all Fourier
                                            ! modes and the melt condition is imposed exactly without
                                            ! iteration) or 'implicit' (theta-method; iterative coupling)
  integer  :: melt_maxit = 50               ! max. iterations of the implicit interface coupling per substep
  real(rp) :: melt_tol = 1.e-12_rp          ! convergence: max change of the interface values (theta, s units)
  !
  ! derived quantities
  !
  integer  :: nxh, mx                      ! number of kx modes kept; physical (dealiased) grid size
  real(rp) :: ybot = 0._rp, ytop = 2._rp   ! wall positions: 0 and ly
  real(rp) :: hgt = 2._rp                  ! wall distance ytop - ybot = ly
  real(rp) :: ymid = 1._rp, hhalf = 1._rp  ! centre and half of the wall distance
  real(rp) :: visc
  real(rp) :: diff(nscal_max)
  real(rp) :: gvec(2) = 0._rp              ! unit vector in the direction of gravity
  logical  :: is_buoyant = .false.         ! at least one active scalar
  logical  :: sxexp(nscal_max) = .false.   ! scalar with explicit x-diffusion (melt_xdiff = 'explicit')
  !
  ! low-storage RK3 of Spalart, Moser & Rogers (1991): nonlinear weights gam, zet and the
  ! implicit/explicit split of the viscous terms (I - b_q dt L) f^{q+1} = (I + a_q dt L) f^q + ...
  !
  real(rp), parameter :: rk_gam(3) = [ 8._rp/15._rp, 5._rp/12._rp,  3._rp/4._rp]
  real(rp), parameter :: rk_zet(3) = [ 0._rp      ,-17._rp/60._rp,-5._rp/12._rp]
  real(rp), parameter :: rk_bet(3) = rk_gam(:) + rk_zet(:)
  real(rp), parameter :: rk_asmr(3) = [29._rp/96._rp, -3._rp/40._rp, 1._rp/6._rp]
  real(rp), parameter :: rk_bsmr(3) = [37._rp/160._rp, 5._rp/24._rp, 1._rp/6._rp]
contains
  subroutine read_input(myid)
    implicit none
    integer, intent(in) :: myid
    integer :: iunit,ios
    namelist /grid/ nx,ny,lx,ly,grid_type,grid_alpha,grid_file,fd_order,dealias
    namelist /physics/ re,forcing,ubulk,dpdx,vv_neumann,nl_form,bc_bot,bc_top
    namelist /time/ nstep,tmax,wall_time_max,dt,fixed_dt,dt_max,dt_growth,cfl,cfl_visc
    namelist /init/ init_type,init_amp,init_seed,mode_file,mode_kx,restart_file,reset_time
    namelist /output/ data_dir,iout_log,iout_stats,iout_save_stats,iout_restart,iout_field,tstats_start, &
                      mode_log_kx,timing
    namelist /scalars/ nscal,pr,scal_theta,sbc_bot,sbc_top,sval_bot,sval_top,ssrc_const,ssrc_u,sinit,scal_form
    namelist /buoyancy/ gravity,ri
    namelist /melt_model/ melt,melt_t,melt_s,melt_st,melt_lambda,melt_thf,melt_sigma,melt_blowing,melt_xdiff, &
                          melt_maxit,melt_tol
    !
    open(newunit=iunit,file='input.nml',status='old',action='read',iostat=ios)
    if(ios /= 0) call abort_run('cannot open input.nml')
    read(iunit,nml=grid    ,iostat=ios); if(ios /= 0) call abort_run('error reading namelist &grid')
    rewind(iunit)
    read(iunit,nml=physics ,iostat=ios); if(ios /= 0) call abort_run('error reading namelist &physics')
    rewind(iunit)
    read(iunit,nml=time    ,iostat=ios); if(ios /= 0) call abort_run('error reading namelist &time')
    rewind(iunit)
    read(iunit,nml=init    ,iostat=ios); if(ios /= 0) call abort_run('error reading namelist &init')
    rewind(iunit)
    read(iunit,nml=output  ,iostat=ios); if(ios /= 0) call abort_run('error reading namelist &output')
    rewind(iunit)
    read(iunit,nml=scalars ,iostat=ios); if(ios > 0 ) call abort_run('error reading namelist &scalars')
    rewind(iunit)
    read(iunit,nml=buoyancy,iostat=ios); if(ios > 0 ) call abort_run('error reading namelist &buoyancy')
    rewind(iunit)
    read(iunit,nml=melt_model,iostat=ios); if(ios > 0 ) call abort_run('error reading namelist &melt_model')
    close(iunit)
    !
    if(mod(nx,2) /= 0 .or. nx < 4) call abort_run('nx must be even and >= 4')
    if(ny < 9) call abort_run('ny must be at least 9')
    if(all(fd_order /= [4,6])) call abort_run('fd_order must be 4 or 6')
    if(dt <= 0._rp) call abort_run('dt must be > 0')
    if(dt_growth < 1._rp) call abort_run('dt_growth must be >= 1')
    if(all(vv_neumann /= ['explicit','compact '])) call abort_run("vv_neumann must be 'explicit' or 'compact'")
    if(any(scal_theta(:) < 0._rp .or. scal_theta(:) > 1._rp)) call abort_run('0 <= scal_theta <= 1 required')
    if(nscal < 0 .or. nscal > nscal_max) call abort_run('0 <= nscal <= 16 required')
    if(any(sbc_bot(1:max(nscal,1)) /= 'D' .and. sbc_bot(1:max(nscal,1)) /= 'N') .or. &
       any(sbc_top(1:max(nscal,1)) /= 'D' .and. sbc_top(1:max(nscal,1)) /= 'N')) &
      call abort_run("sbc_bot and sbc_top must be 'D' or 'N'")
    if(all(forcing /= ['cfr','cpg'])) call abort_run('forcing must be cfr or cpg')
    if(all(nl_form /= ['advective ','rotational'])) call abort_run('nl_form must be advective or rotational')
    if(all(bc_bot /= ['noslip  ','freeslip'])) call abort_run("bc_bot must be 'noslip' or 'freeslip'")
    if(all(bc_top /= ['noslip  ','freeslip'])) call abort_run("bc_top must be 'noslip' or 'freeslip'")
    freeslip_bot = (bc_bot == 'freeslip')
    freeslip_top = (bc_top == 'freeslip')
    if(freeslip_bot .and. freeslip_top) call abort_run('at most one wall can be free-slip (bc_bot, bc_top)')
    if(nscal > 0) then
      if(any(sinit(1:nscal) /= 'linear' .and. sinit(1:nscal) /= 'zero' .and. sinit(1:nscal) /= 'one' .and. &
             sinit(1:nscal) /= 'noise')) call abort_run("sinit must be 'linear', 'zero', 'one' or 'noise'")
    end if
    if(melt) then
      if(freeslip_bot) call abort_run('&melt_model: the ice wall y = 0 must be no-slip (bc_bot)')
      if(melt_t < 1 .or. melt_t > nscal) call abort_run('&melt_model: melt_t must be a scalar index')
      if(sbc_bot(melt_t) /= 'D') call abort_run('&melt_model: sbc_bot of the temperature must be D')
      if(melt_s /= 0) then
        if(melt_s < 1 .or. melt_s > nscal .or. melt_s == melt_t) &
          call abort_run('&melt_model: melt_s must be 0 or another scalar index')
        if(sbc_bot(melt_s) /= 'D') call abort_run('&melt_model: sbc_bot of the salinity must be D')
      end if
      if(melt_st < 0._rp) call abort_run('&melt_model: melt_st must be >= 0')
      if(melt_lambda > 0._rp) call abort_run('&melt_model: melt_lambda must be <= 0')
      if(melt_maxit < 1 .or. melt_tol <= 0._rp) call abort_run('&melt_model: melt_maxit >= 1, melt_tol > 0')
      if(all(melt_xdiff /= ['explicit','implicit'])) call abort_run("&melt_model: melt_xdiff must be 'explicit' or 'implicit'")
      if(melt_xdiff == 'explicit') then
        sxexp(melt_t) = .true.
        if(melt_s > 0) sxexp(melt_s) = .true.
      end if
    end if
    if(all(scal_form /= ['advective ','divergence','skew      '])) &
      call abort_run("scal_form must be 'advective', 'divergence' or 'skew'")
    if(ly <= 0._rp) call abort_run('ly (box height) must be > 0')
    ybot = 0._rp; ytop = ly
    hgt = ytop-ybot; ymid = 0.5_rp*(ybot+ytop); hhalf = 0.5_rp*hgt
    if(trim(grid_type) == 'auto') then
      if(freeslip_bot) then
        grid_type = 'tanh_top'               ! refined at the no-slip wall y = ly only
      else if(freeslip_top) then
        grid_type = 'tanh_bot'               ! refined at the no-slip wall y = 0 only
      else
        grid_type = 'tanh'
      end if
    end if
    if(all(grid_type /= [character(len=16) :: 'tanh','tanh_top','tanh_bot','uniform','file'])) &
      call abort_run("grid_type must be 'auto', 'tanh', 'tanh_top', 'tanh_bot', 'uniform' or 'file'")
    nxh = nx/2
    if(dealias) then
      mx = 3*nx/2
    else
      mx = nx
    end if
    visc = 1._rp/re
    diff(:) = visc/pr(:)
    !
    ! buoyancy
    !
    is_buoyant = any(ri(1:nscal) /= 0._rp)
    if(is_buoyant) then
      if(norm2(gravity) == 0._rp) call abort_run('&buoyancy: gravity must be a non-zero vector')
      gvec(:) = gravity(:)/norm2(gravity)
    end if
    if(myid == 0) call print_setup
  contains
    subroutine print_setup
      integer :: n
      character(len=*), parameter :: f1 = '(1x,a,t36,a)', fi = '(1x,a,t36,3(i0,:,1x))', fr = '(1x,a,t36,4(es12.5,:,1x))'
      write(*,'(a)') ' ============================== solver2d run set-up ==============================='
      write(*,fi) 'Fourier modes nx / points ny',nx,ny
      write(*,fr) 'box length lx',lx
      write(*,fr) 'box height ly (walls at y = 0, ly)',ly
      write(*,f1) 'wall-normal grid',trim(grid_type)
      if(grid_type(1:4) == 'tanh') write(*,fr) '  stretching alpha_s',grid_alpha
      write(*,fi) 'compact scheme order',fd_order
      write(*,f1) '3/2 dealiasing',merge('yes','no ',dealias)
      write(*,fr) 'Reynolds number re',re
      if(forcing == 'cfr') then
        write(*,fr) 'constant flow rate, ubulk',ubulk
      else
        write(*,fr) 'constant pressure gradient -dp/dx',dpdx
      end if
      write(*,f1) 'Navier-Stokes solver','velocity-vorticity (phi = lap v), implicit viscous (SMR RK3)'
      write(*,f1) 'nonlinear term',trim(nl_form)
      write(*,f1) 'wall derivative in dv/dy = 0',trim(vv_neumann)
      write(*,f1) 'wall y = 0',merge('free-slip (v = 0, du/dy = 0)','no-slip                     ',freeslip_bot)
      write(*,f1) 'wall y = ly',merge('free-slip (v = 0, du/dy = 0)','no-slip                     ',freeslip_top)
      if(fixed_dt) then
        write(*,fr) 'fixed time step dt',dt
      else
        write(*,fr) 'adaptive time step, cfl, cfl_visc',cfl,cfl_visc
      end if
      write(*,fi) 'nstep',nstep
      if(tmax < huge(1._rp)) write(*,fr) 'tmax',tmax
      write(*,f1) 'initial condition',trim(init_type)
      write(*,fi) 'number of scalars',nscal
      if(nscal > 0) write(*,f1) 'scalar advection form',trim(scal_form)
      do n=1,nscal
        write(*,'(1x,a,i0,a,es10.3,a,f5.2,a,a1,a,es10.3,a,a1,a,es10.3,a,es10.3)') &
          '  scalar ',n,': Pr = ',pr(n),', theta = ',scal_theta(n),', bottom ',sbc_bot(n),' = ',sval_bot(n), &
          ', top ',sbc_top(n),' = ',sval_top(n),', Ri = ',ri(n)
      end do
      if(is_buoyant) write(*,fr) 'gravity direction (unit vector)',gvec
      if(melt) then
        write(*,'(1x,a,t36,a,i0,a,i0)') 'melting ice at y = 0','temperature: scalar ',melt_t, &
                                        ', salinity: scalar ',melt_s
        write(*,fr) '  St, Lambda, theta_f, sigma',melt_st,melt_lambda,melt_thf,melt_sigma
        write(*,f1) '  meltwater velocity v(0) = m*',merge('yes','no ',melt_blowing)
        if(melt_xdiff == 'explicit') then
          write(*,f1) '  coupling','exact (x-diffusion of theta, s explicit, RK3)'
        else
          write(*,'(1x,a,t36,i0,es12.4)') '  iterative: max. it., tol.',melt_maxit,melt_tol
        end if
      end if
      write(*,'(a)') ' ===================================================================================='
    end subroutine print_setup
  end subroutine read_input
  !
  subroutine abort_run(msg)
    implicit none
    character(len=*), intent(in) :: msg
    integer :: ierr
    write(*,'(a)') 'solver2d ERROR: '//trim(msg)
    flush(6)
    call MPI_ABORT(MPI_COMM_WORLD,1,ierr)
    stop 1
  end subroutine abort_run
end module mod_param
