#!/bin/bash
#PBS -N atm_tissue
#PBS -q base_8
#PBS -l select=1:ncpus=8
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -o /scratch/home/wog3597/ATM/outputs/pbs/
# Atropos 3-class 재실행 -> CSF/GM/WM 확률 + subject 뇌 마스크 (206명).
# 8 워커가 stride 로 나눠 가진다. 산출물 있으면 건너뛴다 (재제출 안전).
set -eo pipefail
cd /scratch/home/wog3597/ATM
mkdir -p outputs/pbs outputs/logs
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1 OMP_NUM_THREADS=1
echo "job $PBS_JOBID on $(hostname) ncpus=$NCPUS start $(date)"
for i in $(seq 0 7); do
  python scripts/69_tissue_extract.py --start $i --stride 8 \
      > outputs/logs/tissue_$i.log 2>&1 &
done
wait
grep -h "TISSUE DONE" outputs/logs/tissue_*.log
echo "ALL TISSUE DONE $(date)"
