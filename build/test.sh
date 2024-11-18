#!/bin/bash

make -j supercell 2>&1 | tee output.txt || exit -1
num_tasks=`echo "$SLURM_NNODES*8" | bc`
srun -N $SLURM_NNODES -n $num_tasks --gpus-per-task=1 --gpu-bind=closest ./supercell  || exit -1
~/spack/opt/spack/linux-sles15-zen/gcc-7.5.0/python-3.11.7-kjzovcrtwquowarhroo5h4gfmcc6nwbe/bin/python postproc/supercell_norms.py  || exit -1
