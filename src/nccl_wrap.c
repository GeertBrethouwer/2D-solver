/*
 * NCCL for the transposes of solver2d (make ... NCCL=1, comm_backend = 'nccl' in &parallel).
 *
 * Called from mod_decomp (bind(c)): the all-to-all of the transposes becomes one group of ncclSend/ncclRecv
 * on the CUDA stream of OpenACC queue 1, the queue of the packing kernels, so no host synchronisation is
 * needed before or after it. NCCL uses NVLink between the GPUs of a node and the network plugin between
 * nodes. MPI still starts the ranks and passes NCCL's unique id (s2d_nccl_get_id -> MPI_BCAST -> s2d_nccl_init).
 *
 * Counts and displacements are in complex(rp) elements, as for MPI_Alltoallv; rp = double precision.
 *
 * S2D_NCCL_MOCK (make NCCL_MOCK=1, CPU build): a test stand-in that does the same sends and receives with
 * non-blocking MPI, to test the NCCL code path of mod_decomp without a GPU. Like NCCL, operations between
 * two ranks are matched in the order they are issued.
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifdef S2D_NCCL_MOCK
#include <mpi.h>
#include <stdlib.h>
#else
#include <cuda_runtime.h>
#include <nccl.h>
static ncclComm_t comm = NULL;
static int report(ncclResult_t r, const char* where) {
  if (r == ncclSuccess) return 0;
  fprintf(stderr, "solver2d: NCCL error in %s: %s (%s)\n", where, ncclGetErrorString(r), ncclGetLastError(comm));
  return 1;
}
#endif

/* NCCL's unique id: 128 bytes, created on rank 0 and broadcast by the caller */
int s2d_nccl_id_size(void) {
#ifdef S2D_NCCL_MOCK
  return 128;
#else
  return (int)sizeof(ncclUniqueId);
#endif
}

int s2d_nccl_get_id(char* id) {
#ifdef S2D_NCCL_MOCK
  memset(id, 0, 128);
  return 0;
#else
  ncclUniqueId u;
  if (report(ncclGetUniqueId(&u), "ncclGetUniqueId")) return 1;
  memcpy(id, &u, sizeof(u));
  return 0;
#endif
}

int s2d_nccl_init(const char* id, int nproc, int rank) {
#ifdef S2D_NCCL_MOCK
  (void)id; (void)nproc; (void)rank;
  return 0;
#else
  ncclUniqueId u;
  memcpy(&u, id, sizeof(u));
  return report(ncclCommInitRank(&comm, nproc, u, rank), "ncclCommInitRank");
#endif
}

int s2d_nccl_finalize(void) {
#ifndef S2D_NCCL_MOCK
  if (comm) { ncclCommDestroy(comm); comm = NULL; }
#endif
  return 0;
}

/* sb, rb: device addresses (host_data use_device); scnt, sdsp, rcnt, rdsp: nproc values, in complex elements */
int s2d_nccl_alltoallv(const double* sb, const int* scnt, const int* sdsp, double* rb, const int* rcnt,
                       const int* rdsp, int nproc, int rank, intptr_t stream) {
#ifdef S2D_NCCL_MOCK
  (void)stream;
  MPI_Request* req = (MPI_Request*)malloc(sizeof(MPI_Request) * 2 * (size_t)nproc);
  int n = 0;
  memcpy(rb + 2 * (size_t)rdsp[rank], sb + 2 * (size_t)sdsp[rank], 16 * (size_t)scnt[rank]);
  for (int p = 0; p < nproc; ++p) {  /* the same order as below: send and receive per peer */
    if (p == rank) continue;
    if (scnt[p] > 0) MPI_Isend(sb + 2 * (size_t)sdsp[p], 2 * scnt[p], MPI_DOUBLE, p, 51, MPI_COMM_WORLD, &req[n++]);
    if (rcnt[p] > 0) MPI_Irecv(rb + 2 * (size_t)rdsp[p], 2 * rcnt[p], MPI_DOUBLE, p, 51, MPI_COMM_WORLD, &req[n++]);
  }
  MPI_Waitall(n, req, MPI_STATUSES_IGNORE);
  free(req);
  return 0;
#else
  cudaStream_t st = (cudaStream_t)stream;
  /* own block: a copy on the same stream */
  if (scnt[rank] > 0 &&
      cudaMemcpyAsync(rb + 2 * (size_t)rdsp[rank], sb + 2 * (size_t)sdsp[rank], 16 * (size_t)scnt[rank],
                      cudaMemcpyDeviceToDevice, st) != cudaSuccess) {
    fprintf(stderr, "solver2d: cudaMemcpyAsync failed in the NCCL transpose\n");
    return 1;
  }
  if (report(ncclGroupStart(), "ncclGroupStart")) return 1;
  for (int p = 0; p < nproc; ++p) {
    if (p == rank) continue;
    if (scnt[p] > 0 &&
        report(ncclSend(sb + 2 * (size_t)sdsp[p], 2 * (size_t)scnt[p], ncclDouble, p, comm, st), "ncclSend")) return 1;
    if (rcnt[p] > 0 &&
        report(ncclRecv(rb + 2 * (size_t)rdsp[p], 2 * (size_t)rcnt[p], ncclDouble, p, comm, st), "ncclRecv")) return 1;
  }
  return report(ncclGroupEnd(), "ncclGroupEnd");
#endif
}
