#!/usr/bin/env bash
# 무인 실행: trimming 판정 -> 백질 학습 기여 분리 -> 하이퍼파라미터 선택 편향 제거
#
# 각 단계는 판정 근거를 남기고, 실패하면 다음 단계로 넘어가지 않는다.
set -eo pipefail
cd /scratch/home/wog3597/ATM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=outputs/logs; mkdir -p $L
W1=outputs/checkpoints/retrain/w1_wm/d3_joint_step1000.pt
J1=outputs/checkpoints/retrain/j1_joint_noanchor/d3_joint_step1000.pt
say() { echo "[$(date +%H:%M)] $*"; }

# A. trimming 판정 -------------------------------------------------------------
say "A. trimming 판정 (w_rl_val vs w_rl_val_trim)"
while pgrep -f "65_rl_alloc.*w_rl_val_trim" >/dev/null; do sleep 60; done
python scripts/83_compare_runs.py w_rl_val w_rl_val_trim | tee $L/a_trim_verdict.log
TRIM_WIN=$(grep '^WINNER=' $L/a_trim_verdict.log | cut -d= -f2)
TARGS=""
[ "$TRIM_WIN" = "w_rl_val_trim" ] && TARGS="--trim --wm-thr 0.5 --gm-margin 12 --max-outside 1.0 --min-len 0"
say "  채택: $TRIM_WIN  (인자: ${TARGS:-없음})"

# B. 백질 학습의 순수 기여 분리 --------------------------------------------------
say "B. 같은 설정을 J1(백질 학습 전) checkpoint 로 -- 백질 학습 기여 분리"
python scripts/65_rl_alloc.py --ckpt "$J1" --modes ridge --alpha 24 --n-probe 64 --iters 100 \
  $TARGS --tag w_rl_val_j1 > $L/b_j1.log 2>&1
python scripts/83_compare_runs.py "$TRIM_WIN" w_rl_val_j1 | tee $L/b_wm_contrib.log

# C. alpha 선택 편향 제거 (train 에서 고르고 val 로 보고) ---------------------------
say "C. alpha 를 train subject 에서 고른다 (val 선택 편향 제거)"
python scripts/81_rl_diag.py --ckpt "$W1" --subjects outputs/splits/train.txt --limit 20 \
  --targets ridge --alphas 8 16 24 32 48 --iters 100 --damps 1.0 --n-probe 64 \
  --tag rl_diag_train > $L/c_alpha_train.log 2>&1
tail -6 $L/c_alpha_train.log
ALPHA_TR=$(python -c "
import json; d=json.load(open('outputs/eval/rl_diag_train.json'))
r=[x for x in d['rows'] if x.get('holdout_resid_r') is not None]
print(int(min(r,key=lambda x: abs(x['holdout_inter']-d['gt_inter']))['alpha']))")
say "  train 이 고른 alpha=$ALPHA_TR"
if [ "$ALPHA_TR" != "24" ]; then
  say "  val 이 고른 24 와 다르다 -> train 선택으로 val 재평가"
  python scripts/65_rl_alloc.py --ckpt "$W1" --modes ridge --alpha $ALPHA_TR --n-probe 64 \
    --iters 100 $TARGS --tag w_rl_val_atr > $L/c_val_atr.log 2>&1
  python scripts/83_compare_runs.py "$TRIM_WIN" w_rl_val_atr | tee $L/c_alpha_verdict.log
else
  say "  train 과 val 이 같은 alpha 를 골랐다 -- 선택 편향 없음"
fi

say "무인 안건 완료"
