#!/bin/bash

source ${MODULESHOME}/init/bash
module reset
module load PrgEnv-gnu cray-parallel-netcdf cmake cray-hdf5 cray-netcdf

unset CXXFLAGS
unset FFLAGS
unset F77FLAGS
unset F90FLAGS

export CC=cc
export FC=ftn
export CXX=CC

unset YAKL_ARCH

unset MPICH_GPU_SUPPORT_ENABLED

export YAKL_CXX_FLAGS="-DMW_GPU_AWARE_MPI -DPORTURB_FUNCTION_TIMERS -O0 -g -Wno-unused-result -Wno-macro-redefined"
export YAKL_F90_FLAGS="-O0 -g -DSCREAM_DOUBLE_PRECISION -ffree-line-length-none"
export MW_LINK_FLAGS=""
export YAKL_DEBUG=ON
export YAKL_PROFILE=ON

unset CXXFLAGS
unset FFLAGS
unset F77FLAGS
unset F90FLAGS

# -I`nc-config --includedir` -I$PNETCDF_DIR/include
# `nc-config --libs` -L$PNETCDF_DIR/lib -lpnetcdf
# -Rpass-analysis=kernel-resource-usage
# -DPORTURB_FUNCTION_TIMERS
# -DYAKL_AUTO_PROFILE
