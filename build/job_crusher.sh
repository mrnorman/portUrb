#!/bin/bash
#SBATCH -A stf006
#SBATCH -J portUrb
#SBATCH -o %x-%j.out
#SBATCH -t 2:00:00
#SBATCH -p batch
#SBATCH -N 64

cd /lustre/orion/stf006/scratch/imn/portUrb/build
source machines/crusher/crusher_gpu.env
srun -n 512 -c 1 --gpus-per-task=1 --gpu-bind=closest ./city ./inputs/input_city.yaml >& portUrb.out

