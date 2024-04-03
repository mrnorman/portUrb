#!/bin/bash
#SBATCH -A stf006
#SBATCH -J portUrb
#SBATCH -o %x-%j.out
#SBATCH -t 0:20:00
#SBATCH -N 2048

num_tasks=`echo "$SLURM_JOB_NUM_NODES*8" | bc`
num_nodes=$SLURM_JOB_NUM_NODES
builddir=/lustre/orion/stf006/scratch/imn/portUrb/build
mydir=$builddir/frontier/nodes_$num_nodes
mkdir -p $mydir
cd $mydir
cp $builddir/wind_farm .
cp $builddir/inputs/input_windfarm_$num_nodes.yaml .
mkdir -p inputs
cp $builddir/inputs/IEA-15-240-RWT.yaml ./inputs
source $builddir/machines/crusher/crusher_gpu.env
srun -n $num_tasks -c 1 --gpus-per-task=1 --gpu-bind=closest ./wind_farm ./input_windfarm_$num_nodes.yaml >& portUrb.out

