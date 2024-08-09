#!/bin/bash
#SBATCH --cluster fawbush
#SBATCH --partition ampere
#SBATCH --cluster-constraint=blue
#SBATCH --exclusive
#SBATCH -A nwp501
#SBATCH -J portUrb
#SBATCH -o %x-%j.out
#SBATCH -t 8:00:00
#SBATCH -N 27

num_tasks=`echo "$SLURM_NNODES*4" | bc`
cd /lustre/storm/nwp501/scratch/imn/portUrb/build
source machines/miller/miller_gpu.env
# srun -N $SLURM_NNODES -n $num_tasks -c 32 --gpus-per-task=1 --gpu-bind=closest ./wind_farm ./inputs/input_windfarm.yaml >& portUrb.out
srun -N $SLURM_NNODES -n $num_tasks -c 32 --gpus-per-task=1 --gpu-bind=closest ./windfarm_ensemble

