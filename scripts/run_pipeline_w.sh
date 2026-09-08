#!/usr/bin/env bash
# W 파이프라인 — 백질 제약 + trimming (기하/전달률 축) + 능선 목표값 + 진폭 (개인차 축)
#
# 왜 두 축인가 (val 31명 실측)
#   전달률 축: RL 역산이 GT 를 목표로 하면 생성 SC 의 resid_r 0.005 -> 0.428 (86배).
#              생성 경로는 개인차를 전달할 수 있다. 백질 제약/trimming 이 이 전달률을 올린다.
#   개인차 축: 목표값 자체의 질. 능선(60 feature) val resid_r 0.153, 선형 진폭 alpha=8 에서
#              subject 간 상관 0.902 (GT 0.904) 를 유지한다. 로그 공간 증폭은 쓰면 안 된다.
#
# 각 단계는 게이트를 통과해야 다음으로 간다. 실패하면 멈춘다 (조용히 나쁜 결과를 넘기지 않는다).
set -eo pipefail
cd /scratch/home/wog3597/ATM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=outputs/logs; mkdir -p $L
say() { echo "[$(date +%H:%M)] $*"; }
need() { [ -f "$1" ] || { echo "없음: $1"; exit 1; }; }

# ---- 1. 스윕 판정 -----------------------------------------------------------
say "1/6 W1 스윕 판정"
python scripts/78_pick_wm_arm.py | tee $L/w_pick_arm.log
WM_GEN=$(grep '^WM_GEN=' $L/w_pick_arm.log | cut -d= -f2)
LR_DEC=$(grep '^LR_DEC=' $L/w_pick_arm.log | cut -d= -f2)
say "  선택: wm_gen=$WM_GEN lr_dec=$LR_DEC"

# ---- 2. 본 학습 -------------------------------------------------------------
say "2/6 W1 본 학습 (1000 step)"
rm -rf outputs/checkpoints/retrain/w1_wm
python scripts/49_d3_joint.py --config configs/retrain/w1_wm.yaml --no-lock \
  --loss-override wm_gen=$WM_GEN --lr-override lr_dec=$LR_DEC \
  --out w1_wm.json > $L/w1_wm.log 2>&1
CK=$(ls -t outputs/checkpoints/retrain/w1_wm/*.pt | head -1); need "$CK"
python -c "
import json,sys; d=json.load(open('outputs/eval/w1_wm.json')); a=d['val']['after']; b=d['val']['before']
print('  recon eval %.3f -> %.3f | valid_conn %.4f -> %.4f'%(b['recon_rmse_eval_mm'],a['recon_rmse_eval_mm'],b['valid_conn'],a['valid_conn']))
assert a['recon_rmse_eval_mm']<=4.6, '게이트 미달: recon %.3f > 4.6'%a['recon_rmse_eval_mm']"
say "  checkpoint: $CK"

# ---- 3. trimming 보정 --------------------------------------------------------
say "3/6 trimming 임계값 보정 (GT 유지율 우선)"
python scripts/73_trim_calibrate.py --ckpt "$CK" --n-subj 4 > $L/w_trim_cal.log 2>&1
python scripts/79_pick_trim.py | tee $L/w_pick_trim.log
TRIM=$(grep '^TRIM=' $L/w_pick_trim.log | cut -d= -f2)
TARGS=""
if [ "$TRIM" = "1" ]; then
  TARGS="--trim --wm-thr $(grep '^WM_THR=' $L/w_pick_trim.log | cut -d= -f2) \
    --gm-margin $(grep '^GM_MARGIN=' $L/w_pick_trim.log | cut -d= -f2) \
    --max-outside $(grep '^MAX_OUTSIDE=' $L/w_pick_trim.log | cut -d= -f2)"
fi
say "  trimming: ${TARGS:-끔}"

# ---- 4. 진폭 보정 + RL 역산 (val) ---------------------------------------------
say "4/6 RL 역산 val 31명 (능선 목표 + GT 오라클 + 기존 배분)"
python scripts/65_rl_alloc.py --ckpt "$CK" --modes ridge,gt --alpha 8 $TARGS \
  --tag w_rl_val > $L/w_rl_val.log 2>&1
need outputs/eval/w_rl_val.json
python -c "
import json; d=json.load(open('outputs/eval/w_rl_val.json'))
print('  %-14s %8s %8s %8s'%('경로','resid_r','inter','abs_r'))
for k in ('base','ridge','gt','target_ridge','target_gt'):
    if k in d: print('  %-14s %8.4f %8.4f %8.4f'%(k,d[k]['resid_r'],d[k]['inter_subj_r'],d[k]['abs_r']))
print('  GT inter %.4f'%d['gt_inter_subj_r'])"

# ---- 5. 진폭 스윕 (subject 간 상관을 GT 에 맞춘다) -------------------------------
say "5/6 진폭 alpha 스윕"
for A in 4 12 20; do
  python scripts/65_rl_alloc.py --ckpt "$CK" --modes ridge --alpha $A $TARGS \
    --tag w_rl_a$A > $L/w_rl_a$A.log 2>&1 || echo "  alpha $A 실패"
done
python scripts/80_alpha_report.py | tee $L/w_alpha.log

say "6/6 W 파이프라인 완료"
