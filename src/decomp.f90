!
! 1D (slab) decomposition and the global transpose
!
!   physical / x-lines  : xa(nxh, nyA)  all kx (after the x-FFT), y-rows yA_lo..yA_lo+nyA-1
!   spectral / y-lines  : f (nkx, ny )  kx = kx_lo..kx_lo+nkx-1, all y (wall-normal solves)
!
! The transpose is one MPI_Alltoallv over all ranks. On GPUs the buffers are device arrays
! (host_data use_device), so a CUDA-aware MPI is required. On one Arrhenius node (4 x GH200)
! the traffic stays on NVLink. A 2D problem has no second direction to decompose, so the
! pencil machinery of HiCHAN (cuDecomp, diezDecomp, process grids) is not needed.
!
module mod_decomp
  use mpi
  use mod_types
  use mod_param, only: nxh,ny,abort_run
#if defined(_OPENACC)
  use openacc
#endif
  implicit none
  private
  public :: init_decomp,finalize_decomp,tr_x2y,tr_y2x
  integer, public :: myid,nproc
  integer, public :: kx_lo,nkx,yA_lo,nyA
  integer(i8), public :: tr_work_bytes = 0
#if defined(_CUDA)
  integer(acc_handle_kind), public :: istream
#endif
  integer, allocatable :: kxlo_r(:),nkx_r(:),ylo_r(:),ny_r(:),own_kx(:),own_y(:)
  integer, allocatable :: sc(:),sd(:),rc(:),rd(:)
  complex(rp), allocatable :: sbuf(:),rbuf(:)
contains
  subroutine init_decomp
    implicit none
    integer :: ierr,p,n
    call MPI_COMM_RANK(MPI_COMM_WORLD,myid ,ierr)
    call MPI_COMM_SIZE(MPI_COMM_WORLD,nproc,ierr)
    if(myid == 0) write(*,'(a,i0)') ' number of MPI processes: ',nproc
#if defined(_CUDA)
    call init_device
    istream = acc_get_cuda_stream(1)
#endif
    if(nproc > nxh .or. nproc > ny) call abort_run('too many MPI processes: need nproc <= nx/2 and nproc <= ny')
    allocate(kxlo_r(0:nproc-1),nkx_r(0:nproc-1),ylo_r(0:nproc-1),ny_r(0:nproc-1))
    do p=0,nproc-1
      call distribute(nxh,nproc,p,kxlo_r(p),nkx_r(p))
      call distribute(ny ,nproc,p,ylo_r(p) ,ny_r(p) )
    end do
    kx_lo = kxlo_r(myid); nkx = nkx_r(myid)
    yA_lo = ylo_r(myid) ; nyA = ny_r(myid)
    allocate(own_kx(nxh),own_y(ny))
    do p=0,nproc-1
      own_kx(kxlo_r(p):kxlo_r(p)+nkx_r(p)-1) = p
      own_y(ylo_r(p):ylo_r(p)+ny_r(p)-1) = p
    end do
    !
    ! x-lines -> y-lines: to rank p go my rows of p's wavenumbers; from rank p come p's rows
    ! of my wavenumbers (the reverse transpose swaps the send and receive counts)
    !
    allocate(sc(0:nproc-1),sd(0:nproc-1),rc(0:nproc-1),rd(0:nproc-1))
    do p=0,nproc-1
      sc(p) = nkx_r(p)*nyA
      rc(p) = nkx*ny_r(p)
    end do
    sd(0) = 0; rd(0) = 0
    do p=1,nproc-1
      sd(p) = sd(p-1)+sc(p-1)
      rd(p) = rd(p-1)+rc(p-1)
    end do
    n = max(nxh*nyA,nkx*ny)
    allocate(sbuf(n),rbuf(n))
    tr_work_bytes = 32_i8*n
    !$acc enter data copyin(kxlo_r,nkx_r,ylo_r,ny_r,own_kx,own_y,sd,rd) create(sbuf,rbuf)
  end subroutine init_decomp
  !
#if defined(_CUDA)
  subroutine init_device
    !
    ! one GPU per rank: node-local rank -> device (as in HiCHAN/CaNS); the assignment is
    ! printed and checked, two ranks must not share a GPU
    !
    use cudafor, only: cudaGetDeviceCount,cudaSetDevice,cudaGetDeviceProperties,cudaDeviceProp
    implicit none
    integer :: local_comm,lrank,mydev,ndev,istat,ierr,hlen,r,q
    integer(acc_device_kind) :: dev_type
    type(cudaDeviceProp) :: prop
    character(len=MPI_MAX_PROCESSOR_NAME) :: host
    character(len=256) :: line
    character(len=256), allocatable :: lines(:)
    character(len=MPI_MAX_PROCESSOR_NAME+16) :: key
    character(len=MPI_MAX_PROCESSOR_NAME+16), allocatable :: keys(:)
    logical :: dup
    call MPI_COMM_SPLIT_TYPE(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,0,MPI_INFO_NULL,local_comm,ierr)
    call MPI_COMM_RANK(local_comm,lrank,ierr)
    istat = cudaGetDeviceCount(ndev)
    if(istat /= 0 .or. ndev < 1) call abort_run('no GPU visible to this MPI rank')
    mydev = mod(lrank,ndev)
    istat = cudaSetDevice(mydev)
    dev_type = acc_get_device_type()
    call acc_set_device_num(mydev,dev_type)
    call MPI_COMM_FREE(local_comm,ierr)
    istat = cudaGetDeviceProperties(prop,mydev)
    call MPI_GET_PROCESSOR_NAME(host,hlen,ierr)
    write(line,'(a,i0,a,a,a,i0,a,i0,a,i0,a,z4.4,a,z2.2,a,z2.2,a)') ' rank ',myid,' on ',trim(host), &
      ': local rank ',lrank,', sees ',ndev,' GPU(s), uses device ',mydev,' (PCI ',prop%pciDomainID,':', &
      prop%pciBusID,':',prop%pciDeviceID,')'
    write(key,'(a,1x,z4.4,z2.2,z2.2)') trim(host),prop%pciDomainID,prop%pciBusID,prop%pciDeviceID
    allocate(lines(0:nproc-1),keys(0:nproc-1))
    call MPI_GATHER(line,len(line),MPI_CHARACTER,lines,len(line),MPI_CHARACTER,0,MPI_COMM_WORLD,ierr)
    call MPI_GATHER(key ,len(key ),MPI_CHARACTER,keys ,len(key ),MPI_CHARACTER,0,MPI_COMM_WORLD,ierr)
    dup = .false.
    if(myid == 0) then
      write(*,'(a)') ' GPU assignment:'
      do r=0,nproc-1
        write(*,'(a)') trim(lines(r))
      end do
      do r=0,nproc-1
        do q=r+1,nproc-1
          if(keys(r) == keys(q)) dup = .true.
        end do
      end do
      flush(6)
    end if
    call MPI_BCAST(dup,1,MPI_LOGICAL,0,MPI_COMM_WORLD,ierr)
    if(dup) call abort_run('several MPI ranks use the same GPU: request one task per GPU '// &
      '(e.g. srun --ntasks-per-node=4 --gpus-per-node=4, without binding all tasks to one device)')
  end subroutine init_device
#endif
  !
  pure subroutine distribute(n,p,c,lo,cnt)
    ! block distribution of n points over p processes (remainder to the last ones)
    implicit none
    integer, intent(in)  :: n,p,c
    integer, intent(out) :: lo,cnt
    integer :: base,nrem
    base = n/p
    nrem = n - base*p
    if(c < p-nrem) then
      cnt = base
      lo  = 1 + c*base
    else
      cnt = base+1
      lo  = 1 + (p-nrem)*base + (c-(p-nrem))*(base+1)
    end if
  end subroutine distribute
  !
  subroutine finalize_decomp
    implicit none
    !$acc exit data delete(kxlo_r,nkx_r,ylo_r,ny_r,own_kx,own_y,sd,rd,sbuf,rbuf)
  end subroutine finalize_decomp
  !
  subroutine tr_x2y(xa,fy)
    ! x-lines (nxh,nyA) -> y-lines (nkx,ny)
    implicit none
    complex(rp), intent(in ) :: xa(nxh,nyA)
    complex(rp), intent(out) :: fy(nkx,ny)
    integer :: i,k,p
    !$acc parallel loop collapse(2) default(present) private(p) async(1)
    do k=1,nyA
      do i=1,nxh
        p = own_kx(i)
        sbuf(sd(p)+1+(i-kxlo_r(p))+nkx_r(p)*(k-1)) = xa(i,k)
      end do
    end do
    call alltoallv(sbuf,sc,sd,rbuf,rc,rd)
    !$acc parallel loop collapse(2) default(present) private(p) async(1)
    do k=1,ny
      do i=1,nkx
        p = own_y(k)
        fy(i,k) = rbuf(rd(p)+1+(i-1)+nkx*(k-ylo_r(p)))
      end do
    end do
  end subroutine tr_x2y
  !
  subroutine tr_y2x(fy,xa)
    ! y-lines (nkx,ny) -> x-lines (nxh,nyA)
    implicit none
    complex(rp), intent(in ) :: fy(nkx,ny)
    complex(rp), intent(out) :: xa(nxh,nyA)
    integer :: i,k,p
    !$acc parallel loop collapse(2) default(present) private(p) async(1)
    do k=1,ny
      do i=1,nkx
        p = own_y(k)
        sbuf(rd(p)+1+(i-1)+nkx*(k-ylo_r(p))) = fy(i,k)
      end do
    end do
    call alltoallv(sbuf,rc,rd,rbuf,sc,sd)
    !$acc parallel loop collapse(2) default(present) private(p) async(1)
    do k=1,nyA
      do i=1,nxh
        p = own_kx(i)
        xa(i,k) = rbuf(sd(p)+1+(i-kxlo_r(p))+nkx_r(p)*(k-1))
      end do
    end do
  end subroutine tr_y2x
  !
  subroutine alltoallv(sb,scnt,sdsp,rb,rcnt,rdsp)
    implicit none
    complex(rp), intent(inout) :: sb(:),rb(:)
    integer, intent(in) :: scnt(:),sdsp(:),rcnt(:),rdsp(:)
    integer :: ierr
    !$acc wait(1)
    !$acc host_data use_device(sb,rb)
    call MPI_ALLTOALLV(sb,scnt,sdsp,MPI_DOUBLE_COMPLEX,rb,rcnt,rdsp,MPI_DOUBLE_COMPLEX,MPI_COMM_WORLD,ierr)
    !$acc end host_data
  end subroutine alltoallv
end module mod_decomp
