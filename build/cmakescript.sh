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

CMAKE_COMMAND=(cmake -Wno-dev -DPORTURB_HOME="`pwd`/.." -DPORTURB_CXX_FLAGS="$PORTURB_CXX_FLAGS")
CMAKE_COMMAND+=(-DPORTURB_CXX_FLAGS="$PORTURB_CXX_FLAGS")
CMAKE_COMMAND+=(-DPORTURB_LINK_FLAGS="$PORTURB_LINK_FLAGS")
CMAKE_COMMAND+=(-DPORTURB_F90_FLAGS="$PORTURB_F90_FLAGS")
[[ ! "$PORTURB_BACKEND" == "" ]] && CMAKE_COMMAND+=(-D${PORTURB_BACKEND}=ON)
[[ ! "$PORTURB_ARCH"    == "" ]] && CMAKE_COMMAND+=(-D${PORTURB_ARCH}=ON)
CMAKE_COMMAND+=($1)

"${CMAKE_COMMAND[@]}"

ln -sf $1/inputs .

