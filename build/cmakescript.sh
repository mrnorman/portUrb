#!/bin/bash

./cmakeclean.sh

if [ $# -ne 1 ]; then
  echo "Error: must pass exactly one parameter giving the directory to build"
  exit -1
fi

if [ ! -d $1 ]; then
  echo "Error: Passed directory does not exist"
  exit -1
fi

if [ ! -f $1/CMakeLists.txt ]; then
  echo "Error: Passed directory does not contain a CMakeLists.txt file"
  exit -1
fi

CMAKE_COMMAND=(cmake)
CMAKE_COMMAND+=(-DYAKL_HAVE_MPI=ON)
CMAKE_COMMAND+=(-Wno-dev)
if [[ "$PORTURB_ARCH" == "serial" ]]; then
  CMAKE_COMMAND+=(-DKokkos_ENABLE_SERIAL=ON)
  CMAKE_COMMAND+=(-DKokkos_ARCH_NATIVE=ON)
fi
if [[ "$PORTURB_ARCH" == "hip" ]]; then
  CMAKE_COMMAND+=(-DKokkos_ENABLE_HIP=ON)
  CMAKE_COMMAND+=(-DKokkos_ARCH_AMD_GFX90A=ON)
fi
if [[ "$PORTURB_DEBUG" == "ON" ]]; then
  CMAKE_COMMAND+=(-DYAKL_DEBUG=ON)
  CMAKE_COMMAND+=(-DKokkos_ENABLE_DEBUG=ON)
  CMAKE_COMMAND+=(-DKokkos_ENABLE_DEBUG_BOUNDS_CHECK=ON)
fi
CMAKE_COMMAND+=($1)
  
ln -sf $1/inputs .

"${CMAKE_COMMAND[@]}"

