!
! optional timers (timing = .true. in &output): wall-clock time of the main parts of the code,
! reported at the end of the run (stdout and data_dir/timing_report.txt) with the min/mean/max
! over the MPI ranks. The timers wait for the GPU (!$acc wait), so asynchronous work is
! attributed to the right part.
!
module mod_timer
  use mpi
  use mod_types
  use mod_param , only: timing,data_dir
  use mod_decomp, only: myid,nproc
  implicit none
  private
  integer, parameter, public :: tm_loop = 1, tm_rk = 2, tm_nl = 3, tm_dt = 4, tm_vsolve = 5, &
                                tm_ssolve = 6, tm_fft = 7, tm_tr = 8, tm_log = 9, tm_stats = 10, &
                                tm_io = 11
  integer, parameter :: ntm = 11
  character(len=44), parameter :: tname(ntm) = [character(len=44) :: &
    'time loop (total)'                           , &
    '  RK3 steps'                                 , &
    '    nonlinear terms (incl. FFTs, buoyancy)'  , &
    '      time-step limit (CFL, MPI_Allreduce)'  , &
    '    velocity solve (vv step)'                , &
    '    scalar solves'                           , &
    'FFTs incl. transposes (all calls)'           , &
    '  transposes (communication, all calls)'     , &
    'log output (divergence, log.dat)'            , &
    'statistics (accumulate, write)'              , &
    'restart and field files'                     ]
  real(rp) :: ttot(ntm) = 0._rp, tbeg(ntm) = 0._rp
  integer  :: ncall(ntm) = 0
  public :: tic,toc,timing_report
contains
  subroutine tic(id)
    integer, intent(in) :: id
    if(.not.timing) return
    !$acc wait
    tbeg(id) = MPI_WTIME()
  end subroutine tic
  !
  subroutine toc(id)
    integer, intent(in) :: id
    if(.not.timing) return
    !$acc wait
    ttot(id) = ttot(id) + (MPI_WTIME()-tbeg(id))
    ncall(id) = ncall(id) + 1
  end subroutine toc
  !
  subroutine timing_report(nsteps)
    integer, intent(in) :: nsteps
    real(rp) :: tmin(ntm),tmax(ntm),tsum(ntm)
    integer :: ierr,n,iunit,u,k
    if(.not.timing) return
    call MPI_REDUCE(ttot,tmin,ntm,MPI_DOUBLE_PRECISION,MPI_MIN,0,MPI_COMM_WORLD,ierr)
    call MPI_REDUCE(ttot,tmax,ntm,MPI_DOUBLE_PRECISION,MPI_MAX,0,MPI_COMM_WORLD,ierr)
    call MPI_REDUCE(ttot,tsum,ntm,MPI_DOUBLE_PRECISION,MPI_SUM,0,MPI_COMM_WORLD,ierr)
    if(myid /= 0) return
    open(newunit=iunit,file=trim(data_dir)//'/timing_report.txt',status='replace')
    do k=1,2
      u = merge(6,iunit,k == 1)
      write(u,'(a)') ' timing report (wall-clock seconds; min/mean/max over the ranks)'
      write(u,'(a44,a10,3a12,a9,a12)') 'part','calls','min','mean','max','% loop','per step'
      do n=1,ntm
        write(u,'(a44,i10,3es12.4,f9.2,es12.4)') tname(n),ncall(n),tmin(n),tsum(n)/nproc,tmax(n), &
          100._rp*tmax(n)/max(tmax(tm_loop),tiny(1._rp)),tmax(n)/max(nsteps,1)
      end do
    end do
    close(iunit)
  end subroutine timing_report
end module mod_timer
