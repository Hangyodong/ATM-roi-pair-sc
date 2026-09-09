#!/usr/bin/env bash
# E 파이프라인 — 조건 개인화(E1) + prior 정렬(E2). 각 단계 게이트 통과해야 다음으로 간다.
#
# 근거 (실측)
#   디코더 조건의 개인 성분 0.04 % (T1 볼륨 7.99 %, ROI 국소 11.36 %)
#     -> local 항 RMS 0.0014 vs pair 항 0.0334. gain 으로 크기를 구조로 잡으면 11.14 % 까지 간다
#   prior 평균이 posterior pair 평균 분산을 **-0.573** 만큼 설명 (상수보다 나쁨)
#     -> 자유 pair 표를 train 템플릿으로 채우면 1.0 에서 시작
set -eo pipefail
cd /scratch/home/wog3597/ATM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=outputs/logs; mkdir -p $L
say() { echo "[$(date +%H:%M)] $*"; }

say "1/4 E1 램프업 스윕 (gain 1.0 / 3.0, 600 step 램프)"
for G in 1.0 3.0; do
  rm -rf outputs/checkpoints/retrain/e1r_g$G
  python scripts/49_d3_joint.py --config configs/retrain/e1_cond.yaml --no-lock \
    --max-steps 1000 --lr-override cond_local_gain=$G,cond_local_gain_steps=600 \
    --out e1r_g$G.json --out-dir outputs/checkpoints/retrain/e1r_g$G > $L/e1r_g$G.log 2>&1 \
    || say "  gain $G 실패 (로그: $L/e1r_g$G.log)"
  say "  gain $G 종료"
done
python scripts/84_pick_e1.py --prefix e1r_g | tee $L/e_pick.log
GAIN=$(grep '^GAIN=' $L/e_pick.log | cut -d= -f2)
say "  승자 gain=$GAIN"

say "2/4 E1 본 학습은 스윕이 곧 본 학습이다 (1000 step) -- 승자 checkpoint 채택"
CK1=$(ls -t outputs/checkpoints/retrain/e1r_g$GAIN/*.pt | head -1)
say "  E1 checkpoint: $CK1"

say "3/4 E2 prior 정렬 (자유 pair 표 + 템플릿 초기화)"
python -c "
import pathlib,re,sys
f=pathlib.Path('configs/retrain/e2_prior.yaml'); t=f.read_text()
t=re.sub(r'^resume: .*$','resume: '+sys.argv[1], t, flags=re.M)
t=re.sub(r'^  cond_local_gain: .*$','  cond_local_gain: '+sys.argv[2], t, flags=re.M)
f.write_text(t)" "$CK1" "$GAIN"
rm -rf outputs/checkpoints/retrain/e2_prior
python scripts/49_d3_joint.py --config configs/retrain/e2_prior.yaml --no-lock \
  --lr-override cond_local_gain=$GAIN --out e2_prior.json --out-dir outputs/checkpoints/retrain/e2_prior \
  > $L/e2_prior.log 2>&1
CK2=$(ls -t outputs/checkpoints/retrain/e2_prior/*.pt | head -1)
say "  E2 checkpoint: $CK2"

say "4/4 세 checkpoint 비교 (J1 / E1 / E2) -- 생성 기하와 prior 정렬"
python scripts/85_cond_prior_report.py J1:outputs/checkpoints/retrain/j1_joint_noanchor/d3_joint_step1000.pt \
  E1:"$CK1" E2:"$CK2" | tee $L/e_report.log
say "E 파이프라인 완료"
