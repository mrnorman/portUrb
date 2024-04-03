#!/bin/bash
#BSUB -P STF006
#BSUB -W 0:20
#BSUB -nnodes 128
#BSUB -J porturb
#BSUB -o porturb.%J
#BSUB -e porturb.%J

num_rs=`echo "($LSB_DJOB_NUMPROC-1)/42*6" | bc`
num_nodes=`echo "($LSB_DJOB_NUMPROC-1)/42" | bc`
builddir=/gpfs/alpine2/stf006/scratch/imn/portUrb/build
mydir=$builddir/summit/nodes_$num_nodes
mkdir -p $mydir
cd $mydir
cp $builddir/wind_farm .
cp $builddir/inputs/input_windfarm_$num_nodes.yaml .
mkdir -p inputs
cp $builddir/inputs/IEA-15-240-RWT.yaml ./inputs
source $builddir/machines/summit/summit_gpu.env
jsrun --smpiargs="-gpu" -n $num_rs -a 1 -c 1 -g 1 ./wind_farm ./input_windfarm_$num_nodes.yaml >& portUrb.out

