#!/bin/bash
#PBS -N atm_route
#PBS -q base_g
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -o /scratch/home/wog3597/ATM/outputs/pbs/
# base_g(H100) 로 ROUTE 파이프라인을 이어받는다. 로컬 A10 실행은 preempt 파일을 보고 스스로 양보한다.
#   qsub scripts/pbs_route.sh
set -uo pipefail
cd /scratch/home/wog3597/ATM
mkdir -p outputs/pbs outputs/checkpoints/route
PRE=outputs/checkpoints/route/preempt
LOCK=outputs/checkpoints/route/pipeline.lock
echo "job $PBS_JOBID on $(hostname) start $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

touch "$PRE"                                  # 1) 로컬에 양보 요청
echo "[pbs] preempt 요청, lock 해제 대기"
for _ in $(seq 120); do                       # 2) 최대 20분 대기
  [ -f "$LOCK" ] || break
  sleep 10
done
[ -f "$LOCK" ] && echo "[pbs] 경고: lock 이 남아 있음 ($(cat $LOCK)) — stale 인수에 맡긴다"
rm -f "$PRE"                                  # 3) 요청 해제 후 내가 잡는다
echo "[pbs] 학습 시작 $(date)"
python scripts/19_train_pipeline.py --config configs/pipeline_route.yaml
code=$?
rm -f "$PRE"
echo "PBS ROUTE DONE code=$code $(date)"
