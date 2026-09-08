#!/usr/bin/env bash
# W2 파이프라인 3~6단계 — 학습된 checkpoint 위에서 trimming 보정 + RL 역산 (튜닝된 설정)
#
# 81_rl_diag 실측 (val 10명, 독립 표본 채점 = 실제 생성에서 나올 값):
#   표본  8, alpha  8 -> 전달률 0.27   실현 resid_r 0.053
#   표본 32, alpha 16 -> 전달률 0.51   실현 0.098
#   표본 64, alpha 24 -> 전달률 0.65   실현 0.122   inter 0.905 (GT 0.878)
# 방문 행렬 Â 의 추정 잡음이 역산에서 함께 증폭되므로 표본 수가 전달률을 지배한다.
set -eo pipefail
cd /scratch/home/wog3597/ATM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=outputs/logs; mkdir -p $L
CK=${1:?checkpoint 경로를 달라}
say() { echo "[$(date +%H:%M)] $*"; }

say "3/6 trimming 임계값 보정 (GT 유지율 >= 0.9 가 절대 조건)"
python scripts/73_trim_calibrate.py --ckpt "$CK" --n-subj 4 > $L/w_trim_cal.log 2>&1
python scripts/79_pick_trim.py | tee $L/w_pick_trim.log
TRIM=$(grep '^TRIM=' $L/w_pick_trim.log | cut -d= -f2)
TARGS=""
if [ "$TRIM" = "1" ]; then
  TARGS="--trim --wm-thr $(grep '^WM_THR=' $L/w_pick_trim.log | cut -d= -f2) --gm-margin $(grep '^GM_MARGIN=' $L/w_pick_trim.log | cut -d= -f2) --max-outside $(grep '^MAX_OUTSIDE=' $L/w_pick_trim.log | cut -d= -f2)"
fi
say "  trimming: ${TARGS:-끔}"

say "4/6 역산 설정 재확인 (새 checkpoint 기하에서 표본/진폭 다시 잰다)"
python scripts/81_rl_diag.py --ckpt "$CK" --limit 10 --targets ridge --alphas 16 24 32 \
  --iters 100 --damps 1.0 --n-probe 64 > $L/w_rl_diag.log 2>&1
tail -4 $L/w_rl_diag.log
ALPHA=$(python -c "
import json; d=json.load(open('outputs/eval/rl_diag.json'))
r=[x for x in d['rows'] if x.get('holdout_resid_r') is not None]
best=min(r,key=lambda x: abs(x['holdout_inter']-d['gt_inter']))
print(int(best['alpha']))")
say "  선택 alpha=$ALPHA (GT subject 간 상관에 가장 가까움)"

say "5/6 RL 역산 val 31명 (능선 목표 alpha=$ALPHA, 표본 64) + GT 오라클"
python scripts/65_rl_alloc.py --ckpt "$CK" --modes ridge,gt --alpha $ALPHA --n-probe 64 \
  --iters 100 $TARGS --tag w_rl_val > $L/w_rl_val.log 2>&1
python -c "
import json; d=json.load(open('outputs/eval/w_rl_val.json'))
print('  %-14s %9s %9s %9s'%('경로','resid_r','inter','abs_r'))
for k in ('base','ridge','gt','target_ridge','target_gt'):
    if k in d: print('  %-14s %9.4f %9.4f %9.4f'%(k,d[k]['resid_r'],d[k]['inter_subj_r'],d[k]['abs_r']))
print('  GT inter %.4f'%d['gt_inter_subj_r'])"

say "6/6 W2 완료"
