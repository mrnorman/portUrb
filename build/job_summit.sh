#!/bin/bash
#BSUB -P stf006
#BSUB -W 6:00
#BSUB -nnodes 46
#BSUB -J portUrb
#BSUB -o portUrb.%J
#BSUB -e portUrb.%J

cd /gpfs/alpine/stf006/scratch/imn/portUrb/build
source machines/summit/summit_gpu.env
date
jsrun -n 276 -r 6 -a 1 -c 1 -g 1 --smpiargs="-gpu" ./driver ./inputs/input_cube.yaml
date

