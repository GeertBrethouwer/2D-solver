#
# solver2d Makefile
#
#   make                       CPU build (gfortran/any MPI wrapper + FFTW3)
#   make GPU=1                 GPU build with nvfortran (OpenACC + cuFFT); needs a CUDA-aware MPI
#   make CONFIG=configs/arrhenius.mk   NSC Arrhenius (GH200)
#   make DEBUG=1               CPU build with run-time checks
#   make ACC_HOST=1            test build: gfortran -fopenacc (OpenACC host fallback, FFTW instead of
#                              cuFFT); checks the directives without a GPU
#   make ... NCCL=1            GPU build: transposes with NCCL (NVLink inside a node; comm_backend in
#                              &parallel). NCCL_HOME is found next to nvcc in the NVIDIA HPC SDK unless set
#   make NCCL_MOCK=1           CPU test build of the NCCL code path (an MPI stand-in for NCCL)
#
CONFIG ?=
ifneq ($(strip $(CONFIG)),)
include $(CONFIG)
endif

ifeq ($(origin FC),default)
FC := mpif90
endif
GPU       ?= 0
ACC_HOST  ?= 0
DEBUG     ?= 0
GPU_CC    ?= cc90
NCCL      ?= 0
NCCL_MOCK ?= 0
CC_NCCL   ?= nvc
FFTW_INC  ?= /usr/include
FFTW_LIB  ?= -lfftw3

SRCDIR   := src
BUILDDIR := build
EXE      := $(BUILDDIR)/solver2d

SRCS := types.f90 param.f90 compact.f90 timer.f90 decomp.f90 fft.f90 ysolve.f90 fields.f90 \
        melt.f90 rk.f90 stats.f90 io.f90 initflow.f90 main.f90
OBJS := $(addprefix $(BUILDDIR)/,$(SRCS:.f90=.o))
ifneq ($(filter 1,$(NCCL) $(NCCL_MOCK)),)
OBJS += $(BUILDDIR)/nccl_wrap.o
endif

LIBS :=
INCS :=

ifeq ($(strip $(GPU)),1)
  # NVIDIA HPC SDK (nvfortran)
  FFLAGS ?= -O3 -fast
  FFLAGS += -cpp -D_CUDA -acc -cuda -gpu=$(GPU_CC) -Minfo=accel -module $(BUILDDIR)
  LIBS   += -cudalib=cufft
else
  ifeq ($(DEBUG),1)
    FFLAGS ?= -O0 -g -fbacktrace -fcheck=all -ffpe-trap=invalid,zero,overflow -Wall -Wno-unused-dummy-argument
  else
    FFLAGS ?= -O3 -march=native
  endif
  FFLAGS += -cpp -ffree-line-length-none -J$(BUILDDIR)
  ifeq ($(ACC_HOST),1)
    FFLAGS += -fopenacc
  endif
  INCS   += -I$(FFTW_INC)
  LIBS   += $(FFTW_LIB)
endif

# NCCL for the transposes (src/nccl_wrap.c, called from mod_decomp)
comma := ,
ifeq ($(NCCL),1)
  NCCL_HOME ?= $(shell sh tools/find_nccl.sh)
  FFLAGS += -D_NCCL
  CFLAGS_NCCL := -O2 -cuda $(if $(strip $(NCCL_HOME)),-I$(NCCL_HOME)/include)
  LIBS   += $(if $(strip $(NCCL_HOME)),-L$(NCCL_HOME)/lib -Wl$(comma)-rpath$(comma)$(NCCL_HOME)/lib) -lnccl
else ifeq ($(NCCL_MOCK),1)
  FFLAGS += -D_NCCL
  CC_NCCL := mpicc
  CFLAGS_NCCL := -O2 -DS2D_NCCL_MOCK
endif

.PHONY: all clean
all: $(EXE)

$(BUILDDIR)/nccl_wrap.o: $(SRCDIR)/nccl_wrap.c | $(BUILDDIR)
	$(CC_NCCL) $(CFLAGS_NCCL) -c $< -o $@

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

$(EXE): $(OBJS)
	$(FC) $(FFLAGS) $(OBJS) $(LIBS) -o $@

$(BUILDDIR)/%.o: $(SRCDIR)/%.f90 | $(BUILDDIR)
	$(FC) $(FFLAGS) $(INCS) -c $< -o $@

# module dependencies
$(BUILDDIR)/param.o:    $(BUILDDIR)/types.o
$(BUILDDIR)/compact.o:  $(BUILDDIR)/param.o
$(BUILDDIR)/timer.o:    $(BUILDDIR)/param.o
$(BUILDDIR)/decomp.o:   $(BUILDDIR)/param.o $(BUILDDIR)/timer.o
$(BUILDDIR)/fft.o:      $(BUILDDIR)/decomp.o $(BUILDDIR)/timer.o
$(BUILDDIR)/ysolve.o:   $(BUILDDIR)/decomp.o $(BUILDDIR)/compact.o
$(BUILDDIR)/fields.o:   $(BUILDDIR)/decomp.o
$(BUILDDIR)/melt.o:     $(BUILDDIR)/fft.o $(BUILDDIR)/ysolve.o
$(BUILDDIR)/rk.o:       $(BUILDDIR)/fields.o $(BUILDDIR)/fft.o $(BUILDDIR)/ysolve.o $(BUILDDIR)/timer.o $(BUILDDIR)/melt.o
$(BUILDDIR)/stats.o:    $(BUILDDIR)/ysolve.o $(BUILDDIR)/melt.o
$(BUILDDIR)/io.o:       $(BUILDDIR)/decomp.o
$(BUILDDIR)/initflow.o: $(BUILDDIR)/ysolve.o $(BUILDDIR)/fields.o
$(BUILDDIR)/main.o:     $(BUILDDIR)/melt.o $(BUILDDIR)/timer.o $(BUILDDIR)/rk.o $(BUILDDIR)/stats.o $(BUILDDIR)/io.o $(BUILDDIR)/initflow.o

clean:
	rm -rf $(BUILDDIR)
