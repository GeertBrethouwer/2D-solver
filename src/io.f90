!
! restart files (spectral, MPI-IO) and 2D field output (physical, MPI-IO + XDMF)
!
module mod_io
  use mpi
  use mod_types
  use mod_param
  use mod_decomp, only: myid,nkx,kx_lo,nyA,yA_lo
  implicit none
  private
  public :: write_restart,read_restart,write_field,write_xdmf
contains
  subroutine spec_subarray(dtype)
    implicit none
    integer, intent(out) :: dtype
    integer :: ierr
    call MPI_TYPE_CREATE_SUBARRAY(2,[nxh,ny],[nkx,ny],[kx_lo-1,0], &
                                  MPI_ORDER_FORTRAN,MPI_DOUBLE_COMPLEX,dtype,ierr)
    call MPI_TYPE_COMMIT(dtype,ierr)
  end subroutine spec_subarray
  !
  subroutine write_restart(fname,time,istep,dtcur,u,v,s)
    !
    ! file layout: u, v, s_1, ..., s_nscal as global complex arrays (nx/2, ny), independent of the
    ! number of ranks; a small text header <fname>.hdr holds the metadata
    !
    implicit none
    character(len=*), intent(in) :: fname
    real(rp), intent(in) :: time,dtcur
    integer , intent(in) :: istep
    complex(rp), intent(in) :: u(nkx,ny),v(nkx,ny)
    complex(rp), intent(in), contiguous :: s(:,:,:)
    integer :: fh,dtype,ierr,n,iunit
    integer(MPI_OFFSET_KIND) :: disp,fsize
    !$acc wait(1)
    !$acc update self(u,v,s)
    call spec_subarray(dtype)
    call MPI_FILE_OPEN(MPI_COMM_WORLD,trim(fname)//'.bin',MPI_MODE_CREATE+MPI_MODE_WRONLY,MPI_INFO_NULL,fh,ierr)
    if(ierr /= MPI_SUCCESS) call abort_run('cannot open '//trim(fname)//'.bin for writing')
    fsize = 0
    call MPI_FILE_SET_SIZE(fh,fsize,ierr)
    fsize = int(nxh,MPI_OFFSET_KIND)*ny*16
    disp = 0
    call write_one(u)
    call write_one(v)
    do n=1,nscal
      call write_one(s(:,:,n))
    end do
    call MPI_FILE_CLOSE(fh,ierr)
    call MPI_TYPE_FREE(dtype,ierr)
    if(myid == 0) then
      open(newunit=iunit,file=trim(fname)//'.hdr',status='replace',action='write')
      write(iunit,'(a)') '# nx ny nscal / lx ybot ytop / time istep dt / grid_type grid_alpha fd_order'
      write(iunit,*) nx,ny,nscal
      write(iunit,*) lx,ybot,ytop
      write(iunit,*) time,istep,dtcur
      write(iunit,*) trim(grid_type),grid_alpha,fd_order
      close(iunit)
    end if
  contains
    subroutine write_one(f)
      complex(rp), intent(in) :: f(nkx,ny)
      call MPI_FILE_SET_VIEW(fh,disp,MPI_DOUBLE_COMPLEX,dtype,'native',MPI_INFO_NULL,ierr)
      call MPI_FILE_WRITE_ALL(fh,f,nkx*ny,MPI_DOUBLE_COMPLEX,MPI_STATUS_IGNORE,ierr)
      disp = disp + fsize
    end subroutine write_one
  end subroutine write_restart
  !
  subroutine read_restart(fname,time,istep,dtcur,u,v,s)
    implicit none
    character(len=*), intent(in) :: fname
    real(rp), intent(out) :: time,dtcur
    integer , intent(out) :: istep
    complex(rp), intent(out) :: u(nkx,ny),v(nkx,ny)
    complex(rp), intent(inout), contiguous :: s(:,:,:)
    integer :: fh,dtype,ierr,n,iunit,ios,nxr,nyr,nsr
    integer(MPI_OFFSET_KIND) :: disp,fsize
    real(rp) :: lxr,ybr,ytr
    character(len=512) :: line
    open(newunit=iunit,file=trim(fname)//'.hdr',status='old',action='read',iostat=ios)
    if(ios /= 0) call abort_run('cannot open restart header '//trim(fname)//'.hdr')
    read(iunit,*)
    read(iunit,*) nxr,nyr,nsr
    read(iunit,'(a)') line
    read(line,*,iostat=ios) lxr,ybr,ytr
    if(ios /= 0) then           ! older files: lx only, walls at -1 and 1
      read(line,*) lxr
      ybr = -1._rp; ytr = 1._rp
    end if
    read(iunit,*) time,istep,dtcur
    close(iunit)
    if(nxr /= nx .or. nyr /= ny) call abort_run('restart file has a different resolution')
    if(abs(lxr-lx) > 1.e-10_rp*lx) call abort_run('restart file has a different box length')
    ! only the box height matters (files of older versions have their walls at -1 and 1)
    if(abs((ytr-ybr)-ly) > 1.e-10_rp*ly) call abort_run('restart file has a different box height ly')
    call spec_subarray(dtype)
    call MPI_FILE_OPEN(MPI_COMM_WORLD,trim(fname)//'.bin',MPI_MODE_RDONLY,MPI_INFO_NULL,fh,ierr)
    if(ierr /= MPI_SUCCESS) call abort_run('cannot open '//trim(fname)//'.bin')
    fsize = int(nxh,MPI_OFFSET_KIND)*ny*16
    disp = 0
    call read_one(u)
    call read_one(v)
    do n=1,nscal
      if(n <= nsr) then
        call read_one(s(:,:,n))
      else
        if(myid == 0) write(*,'(a,i0,a)') ' scalar ',n,' not in the restart file: initialised from input'
      end if
    end do
    call MPI_FILE_CLOSE(fh,ierr)
    call MPI_TYPE_FREE(dtype,ierr)
    !$acc update device(u,v,s)
  contains
    subroutine read_one(f)
      complex(rp), intent(out) :: f(nkx,ny)
      call MPI_FILE_SET_VIEW(fh,disp,MPI_DOUBLE_COMPLEX,dtype,'native',MPI_INFO_NULL,ierr)
      call MPI_FILE_READ_ALL(fh,f,nkx*ny,MPI_DOUBLE_COMPLEX,MPI_STATUS_IGNORE,ierr)
      disp = disp + fsize
    end subroutine read_one
  end subroutine read_restart
  !
  subroutine write_field(istep,name,f)
    !
    ! one physical field on the (dealiased) mx x ny grid, Fortran order (x,y), raw binary
    !
    implicit none
    integer , intent(in) :: istep
    character(len=*), intent(in) :: name
    real(rp), intent(in) :: f(mx,nyA)
    integer :: fh,dtype,ierr
    integer(MPI_OFFSET_KIND) :: disp
    character(len=7) :: cstep
    write(cstep,'(i7.7)') istep
    !$acc wait(1)
    !$acc update self(f)
    call MPI_TYPE_CREATE_SUBARRAY(2,[mx,ny],[mx,nyA],[0,yA_lo-1], &
                                  MPI_ORDER_FORTRAN,MPI_DOUBLE_PRECISION,dtype,ierr)
    call MPI_TYPE_COMMIT(dtype,ierr)
    call MPI_FILE_OPEN(MPI_COMM_WORLD,trim(data_dir)//'/fld_'//cstep//'_'//trim(name)//'.bin', &
                       MPI_MODE_CREATE+MPI_MODE_WRONLY,MPI_INFO_NULL,fh,ierr)
    disp = 0
    call MPI_FILE_SET_SIZE(fh,disp,ierr)
    call MPI_FILE_SET_VIEW(fh,disp,MPI_DOUBLE_PRECISION,dtype,'native',MPI_INFO_NULL,ierr)
    call MPI_FILE_WRITE_ALL(fh,f,mx*nyA,MPI_DOUBLE_PRECISION,MPI_STATUS_IGNORE,ierr)
    call MPI_FILE_CLOSE(fh,ierr)
    call MPI_TYPE_FREE(dtype,ierr)
  end subroutine write_field
  !
  subroutine write_xdmf(istep,time,names,yg)
    ! XDMF descriptor (ParaView/VisIt): X = streamwise, Y = wall-normal
    implicit none
    integer , intent(in) :: istep
    real(rp), intent(in) :: time,yg(ny)
    character(len=*), intent(in) :: names(:)
    integer :: iunit,i,n
    character(len=7) :: cstep
    if(myid /= 0) return
    write(cstep,'(i7.7)') istep
    open(newunit=iunit,file=trim(data_dir)//'/grid_x.bin',access='stream',status='replace')
    write(iunit) [(lx*(i-1)/mx,i=1,mx)]
    close(iunit)
    open(newunit=iunit,file=trim(data_dir)//'/grid_y.bin',access='stream',status='replace')
    write(iunit) yg(:)
    close(iunit)
    open(newunit=iunit,file=trim(data_dir)//'/fld_'//cstep//'.xmf',status='replace')
    write(iunit,'(a)') '<?xml version="1.0" ?>'
    write(iunit,'(a)') '<Xdmf Version="2.0">'
    write(iunit,'(a)') ' <Domain>'
    write(iunit,'(a)') '  <Grid Name="channel2d" GridType="Uniform">'
    write(iunit,'(a,es16.8,a)') '   <Time Value="',time,'"/>'
    write(iunit,'(a,2(i0,1x),a)') '   <Topology TopologyType="2DRectMesh" Dimensions="',ny,mx,'"/>'
    write(iunit,'(a)') '   <Geometry GeometryType="VXVY">'
    call xdmf_item(iunit,'grid_x.bin',[mx])
    call xdmf_item(iunit,'grid_y.bin',[ny])
    write(iunit,'(a)') '   </Geometry>'
    do n=1,size(names)
      write(iunit,'(a)') '   <Attribute Name="'//trim(names(n))//'" AttributeType="Scalar" Center="Node">'
      call xdmf_item(iunit,'fld_'//cstep//'_'//trim(names(n))//'.bin',[ny,mx])
      write(iunit,'(a)') '   </Attribute>'
    end do
    write(iunit,'(a)') '  </Grid>'
    write(iunit,'(a)') ' </Domain>'
    write(iunit,'(a)') '</Xdmf>'
    close(iunit)
  end subroutine write_xdmf
  !
  subroutine xdmf_item(iunit,fname,dims)
    implicit none
    integer, intent(in) :: iunit,dims(:)
    character(len=*), intent(in) :: fname
    character(len=64) :: cd
    integer :: i
    cd = ''
    do i=1,size(dims)
      write(cd,'(a,1x,i0)') trim(cd),dims(i)
    end do
    write(iunit,'(a)') '    <DataItem Dimensions="'//trim(adjustl(cd))// &
                       '" NumberType="Float" Precision="8" Format="Binary">'
    write(iunit,'(a)') '     '//trim(fname)
    write(iunit,'(a)') '    </DataItem>'
  end subroutine xdmf_item
end module mod_io
