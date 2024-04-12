#!/bin/bash
#BSUB -P STF006
#BSUB -W 8:00
#BSUB -nnodes 512
#BSUB -J porturb
#BSUB -o porturb.%J
#BSUB -e porturb.%J

num_rs=`echo "($LSB_DJOB_NUMPROC-1)/42*6" | bc`
num_nodes=`echo "($LSB_DJOB_NUMPROC-1)/42" | bc`
cd /gpfs/alpine2/stf006/scratch/imn/portUrb/build
source machines/summit/summit_gpu.env
jsrun --smpiargs="-gpu" -n $num_rs -a 1 -c 1 -g 1 ./turbine_hires ./inputs/input_turbine_hires.yaml >& portUrb.out

