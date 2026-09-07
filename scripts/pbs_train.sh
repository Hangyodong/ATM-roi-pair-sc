#!/bin/bash
#PBS -N atm_sc_train
#PBS -q base_g
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -o /scratch/home/wog3597/ATM/outputs/pbs/
# 사용: qsub -v PIPELINE=configs/pipeline.yaml scripts/pbs_train.sh   (9-phase, 이어받기)
#       qsub -v CONFIG=configs/phase2_geometry.yaml scripts/pbs_train.sh  (단일 phase)
# base_g 는 사용자당 running GPU 1개 제한 (max_run_res.ngpus=[u:PBS_GENERIC=1]).
# interactive GPU 세션이 떠 있으면 이 job 은 그 세션이 끝날 때까지 Q 상태로 기다린다.
set -eo pipefail
cd /scratch/home/wog3597/ATM
mkdir -p outputs/pbs
export OMP_NUM_THREADS=${NCPUS:-8} PYTHONWARNINGS=ignore
echo "job $PBS_JOBID on $(hostname) config=$CONFIG start $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [ -n "$PIPELINE" ]; then
  # 로컬(A10)에서 돌던 학습을 pipeline_state.json / *_latest.pt 에서 이어받는다
  python scripts/19_train_pipeline.py --config "$PIPELINE"
else
  python scripts/16_train_config.py --config "$CONFIG"
fi
echo "end $(date)"
