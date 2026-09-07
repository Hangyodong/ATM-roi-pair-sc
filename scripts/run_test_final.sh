#!/usr/bin/env bash
# 최종 test 1회. 사용: bash scripts/run_test_final.sh <checkpoint> <resid_gain>
# test 는 한 번만 쓴다 (memory: test-split-used-once). 결과를 보고 다시 튜닝하지 않는다.
set -eu
cd /scratch/home/wog3597/ATM
CK=$1; GAIN=${2:-8}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[$(date +%H:%M)] test 평가 시작: $CK gain=$GAIN"
python scripts/49_eval_frozen.py --ckpt "$CK" --subjects outputs/splits/test.txt \
  --by-count --resid-alloc --resid-gain "$GAIN" --n-per-pair 16 --total-streamlines 460000 \
  > outputs/eval/test_final.log 2>&1
echo "[$(date +%H:%M)] test 평가 종료: $?"
V=outputs/eval/final_$(basename "$CK" .pt)_vectors.npz
python scripts/62_sc_figs.py --vectors "$V" --pred-key pred_best --pred-label "generated (alloc)" --per-fig 8
python scripts/62_sc_figs.py --vectors "$V" --pred-key pred_generated --pred-label "generated (uniform)" --per-fig 8
echo "[$(date +%H:%M)] 그림 완료"
