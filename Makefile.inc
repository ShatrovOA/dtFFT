#####################################################################################
#  Start of user-defined options
#####################################################################################
# Choose compiler
# Only one option should be provided
# Possible options are:
# - intel
# - gnu
# - nvidia
COMPILER      = gnu

# Choose external fft libs
# Several options can be provided
# Possible options are:
# - fftw3
# - mkl
# - cufft
EXTERNAL_LIBS = fftw3

# Choose build mode
# Only one option should be provided
# Possible options are:
# - debug
# - release
# - coverage
BUILD_MODE = coverage

# Choose library type
# Several options can be provided
# Possible options are:
# - static
# - shared
LIB_TYPE = static

# Set path to FFTW3 root directory
FFTW3_DIR = ${HOME}/fftw3

# Set path to Nvidia Open-MPI bin directory
# Make sure that pgfortran or nvfortran is in your PATH
NVIDIA_MPI_BIN   = 
#####################################################################################
#  End of user-defined options
#####################################################################################

BUILD_DIR   = build
OBJ_DIR     = $(BUILD_DIR)/obj/
MOD_DIR     = $(BUILD_DIR)/mod/
STATIC_LIB  = $(BUILD_DIR)/libdtfft.a
SHARED_LIB  = $(BUILD_DIR)/libdtfft.so
EXE_DIR     = exe


ifneq ($(filter $(BUILD_MODE),debug coverage),)
  OPTIMIZE = -O0 -g
else 
  OPTIMIZE = -O3
endif

MPI_LIB = 
COMPILER_FLAGS = 
PREPROC = 

ifeq ($(BUILD_MODE), debug)
  PREPROC += -D__DEBUG
endif

ifeq ($(COMPILER), intel)
  FC      = mpiifort
  CC      = mpiicc
  MOD     = -module
  FLIBS   = -lifcore -lifcoremt
  ifeq ($(BUILD_MODE), debug)
    COMPILER_FLAGS +=-warn all -check all -ftrapuv -fpe3 -assume ieee_fpe_flags -warn interfaces
  endif
else ifeq ($(COMPILER), gnu)
  FC      = mpifort
  CC      = mpicc
  MOD     = -J
  FLIBS   = -lgfortran
  MPI_LIB = -lmpi_usempif08
  COMPILER_FLAGS += -m64 -ffree-line-length-none
  ifeq ($(BUILD_MODE), coverage)
    COMPILER_FLAGS += -fprofile-arcs -ftest-coverage
  else ifeq ($(BUILD_MODE), debug)
    COMPILER_FLAGS += -W -Wall -ggdb -fopt-info -fopt-info-optall-optimized -ftree-vectorize -Wline-truncation    \
                      -Wcharacter-truncation -Wsurprising -Waliasing -Wimplicit-interface -Wunused-parameter      \
                      -fwhole-file  -fcheck=all -ffpe-trap=invalid,zero,overflow,underflow -ffpe-summary=none     \
                      -Warray-bounds -Wimplicit-procedure -Wunderflow -Wuninitialized -fimplicit-none -fbacktrace \
                      -fdump-core -finit-real=nan
  endif
else ifeq ($(COMPILER), nvidia)
  FC      = $(NVIDIA_MPI_BIN)/mpifort
  CC      = $(NVIDIA_MPI_BIN)/mpicc
  MOD     = -module
  FLIBS   = -pgf90libs
  COMPILER_FLAGS += -acc -Mcudalib=cufft -ta=tesla:cc70 -Minfo=accel
  ifeq ($(BUILD_MODE), release)
    OPTIMIZE += -fast
  endif
  MPI_LIB = -lmpi_usempif08
endif


STATIC_LIB_COMMAND = 
ifeq ($(findstring static, $(LIB_TYPE)), static)
  STATIC_LIB_COMMAND = ar rcs $(STATIC_LIB) $(OBJ_DIR)*.o
endif

MKL_INCLUDE = 
MKL_LINK    =
ifeq ($(findstring mkl, $(EXTERNAL_LIBS)), mkl)
  PREPROC += -DMKL_ENABLED
  MKL_INCLUDE = -I${MKLROOT}/include 
  ifeq ($(COMPILER), intel)
    MKL_LINK = -L${MKLROOT}/lib -Wl,-rpath,${MKLROOT}/lib -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
  else ifeq ($(COMPILER), gnu)
    MKL_LINK = -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_gf_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
  else ifeq ($(COMPILER), nvidia)
    MKL_LINK = -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
  endif
endif

FFTW3_INCLUDE = -I $(FFTW3_DIR)/include
FFTW3_LINK    = -L $(FFTW3_DIR)/lib -lfftw3 -lfftw3f 
ifeq ($(findstring fftw3, $(EXTERNAL_LIBS)),)
  FFTW3_INCLUDE = 
  FFTW3_LINK    =
  PREPROC       += -DNO_FFTW3
endif

CUFFT_LINK    =
ifeq ($(findstring cufft, $(EXTERNAL_LIBS)), cufft)
  CUFFT_LINK     = -acc -Mcudalib=cufft -ta=tesla:cc70 -Minfo=accel
  PREPROC        += -DCUFFT_ENABLED
endif

INCLUDES = $(MKL_INCLUDE) $(FFTW3_INCLUDE)

COMMON_FLAGS  = $(PREPROC) $(OPTIMIZE)
LINK_DTFFT    = ../../$(STATIC_LIB)
LINK_EXTERNAL = $(FFTW3_LINK) $(MKL_LINK) $(CUFFT_LINK)

ifeq ($(BUILD_MODE), coverage)
  LINK_EXTERNAL += -lgcov --coverage
endif

SHARED_LIB_COMMAND =
ifeq ($(findstring shared, $(LIB_TYPE)), shared)
  SHARED_LIB_COMMAND = $(FC) -shared -o $(SHARED_LIB) $(OBJ_DIR)*.o $(LINK_EXTERNAL)
  COMPILER_FLAGS += -fPIC
endif

FC_BUILD_FLAGS  = $(COMMON_FLAGS) $(COMPILER_FLAGS) $(MOD) ../$(MOD_DIR) $(INCLUDES)
FC_LINK_FLAGS   = $(COMMON_FLAGS) $(FFTW3_INCLUDE) $(COMPILER_FLAGS) -I ../../$(MOD_DIR) $(LINK_DTFFT) $(LINK_EXTERNAL)

CC_BUILD_FLAGS  = $(COMMON_FLAGS)
CC_LINK_FLAGS   = $(COMMON_FLAGS) $(FFTW3_INCLUDE) -I ../../include $(LINK_DTFFT) $(LINK_EXTERNAL) $(FLIBS) $(MPI_LIB)

COTEXT  = "Compiling $(<F)"
LITEXT  = "Assembling dtFFT"
BUTEXT  = "Building $@"