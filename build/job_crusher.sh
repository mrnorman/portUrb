#!/bin/bash
#SBATCH -A stf006
#SBATCH -J portUrb
#SBATCH -o %x-%j.out
#SBATCH -t 2:00:00
#SBATCH -N 100

num_nodes=`echo "$SLURM_JOB_NUM_NODES*8" | bc`
cd /lustre/orion/stf006/scratch/imn/portUrb/build
source machines/crusher/crusher_gpu.env
srun -n $num_nodes -c 1 --gpus-per-task=1 --gpu-bind=closest ./abl ./inputs/input_abl_neutral.yaml >& portUrb.out

