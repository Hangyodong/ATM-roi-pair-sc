#!/usr/bin/env bash
# synthetic cache 가 끝나면 base_g(H100) job 을 제출한다. job 이 잡히면 pbs_route.sh 가 preempt 파일로
# 로컬 A10 실행에 양보를 요청하고 latest checkpoint 에서 이어받는다.
#   nohup setsid bash scripts/28_submit_h100.sh > outputs/pbs/submit_h100.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
mkdir -p outputs/pbs
echo "[h100] $(date) synthetic cache 완료 대기"
while [ ! -f outputs/synthetic/augment_summary.json ]; do sleep 60; done
echo "[h100] $(date) cache 완료, base_g 제출"
JOB=$(qsub scripts/pbs_route.sh)
echo "$JOB" > outputs/pbs/h100_job_id.txt
echo "[h100] job $JOB 제출됨"
while true; do                       # 상태를 30분마다 남긴다
  s=$(qstat -f "$JOB" 2>/dev/null | awk -F' = ' '/job_state/{print $2}')
  [ -z "$s" ] && { echo "[h100] $(date) job 종료/소멸"; break; }
  echo "[h100] $(date) job $JOB state=$s"
  [ "$s" = "R" ] && { echo "[h100] 실행 시작 — 로컬은 preempt 로 양보"; break; }
  sleep 1800
done
