#!/bin/bash
#PBS -N atm_sc_preproc
#PBS -q base_8
#PBS -l select=1:ncpus=8
#PBS -l walltime=06:00:00
#PBS -j oe
#PBS -o /scratch/home/wog3597/ATM/outputs/pbs/
# CPU 전처리 (01 SyN -> 02 ROI-pair -> 03 bundle). 8 subject 동시 x ITK 4 스레드.
# 로컬 배치와 반대 방향(--reverse)으로 처리해 충돌 없이 중간에서 만난다. 산출물 있으면 건너뜀.
set -eo pipefail
cd /scratch/home/wog3597/ATM
mkdir -p outputs/pbs
echo "job $PBS_JOBID on $(hostname) ncpus=$NCPUS start $(date)"
python scripts/12_preprocess_batch.py --workers 4 --threads-per-worker 2 --reverse
echo "PREPROC DONE $(date)"
