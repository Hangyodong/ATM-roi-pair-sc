#!/usr/bin/env bash
# 1~5 단계를 순서대로. GPU 는 한 번에 하나씩 (full UNet 이 15.9GB 라 병렬 불가).
set -u
cd /scratch/home/wog3597/ATM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=outputs/eval
log(){ echo "[$(date +%H:%M)] $*" | tee -a $L/pipeline.log; }
# 단계가 실패하면 **여기서 멈춘다**. 전에는 A3 가 죽어도 D4 로 넘어가 삭제된 경로를 집었다 (20:20).
need_ok(){ rc=$1; shift; if [ "$rc" -ne 0 ]; then log "실패(rc=$rc): $* -- 파이프라인 중단"; exit "$rc"; fi; }
need_ck(){ if [ -z "$1" ] || [ ! -s "$1" ]; then log "checkpoint 없음: '$1' -- 파이프라인 중단"; exit 2; fi; }

log "2차 파이프라인: D-f''' (중심화 prior) -> D4 -> J1 -> A/B. A3 는 완료됨"

# --- 3) prior 재학습 (pair 별 저랭크, E8 위에서) -----------------------------
log "D-f (subject 조건부 prior) 시작"
python scripts/49_d3_joint.py --config configs/retrain/df_prior.yaml --no-lock > $L/df_prior.log 2>&1
rc=$?; log "D-f 종료: $rc"; need_ok $rc D-f
DF=$(ls -t outputs/checkpoints/retrain/df_prior/*.pt 2>/dev/null | head -1)
need_ck "$DF"; log "D-f checkpoint = $DF"


# D-f 판정 (subject 조건부 prior 가 됐는가) -- 실패면 여기서 멈춘다
python scripts/63_prior_subject_share.py --ckpt "$DF" 2>/dev/null | sed -n '/^{/,$p' > $L/df_judge.json
SHARE=$(python -c "import json;print(json.load(open('$L/df_judge.json'))['prior_mu_subject_share'])")
log "D-f 판정: prior mu subject 성분 = $SHARE (게이트 > 0.005, 목표 0.05)"
python -c "import sys; sys.exit(0 if float(sys.argv[1]) > 0.005 else 1)" "$SHARE"; need_ok $? 'D-f 판정(subject 성분 <= 0.005)'

# --- 5) 디코더 재학습 (좌표 상자 / 앵커) -------------------------------------
log "D-a/b/c 디코더+앵커 시작"
CK5="$DF"
python -c "import pathlib,re,sys; f=pathlib.Path('configs/retrain/d4_anchor.yaml'); f.write_text(re.sub(r'^resume: .*$', 'resume: '+sys.argv[1], f.read_text(), flags=re.M))" "$CK5"
python scripts/57_d4_anchor.py --config configs/retrain/d4_anchor.yaml > $L/d4_anchor2.log 2>&1
rc=$?; log "D4 종료: $rc"; need_ok $rc D4

D4=$(python -c "import json;print(json.load(open('outputs/eval/d4_anchor.json'))['checkpoint'])" 2>/dev/null)
need_ck "$D4"; log "D4 checkpoint = $D4"

# --- 6) joint fine-tune (D4 위에서, 전부 낮은 LR) ---------------------------
python -c "import pathlib,re,sys; f=pathlib.Path('configs/retrain/j1_joint.yaml'); f.write_text(re.sub(r'^resume: .*$', 'resume: '+sys.argv[1], f.read_text(), flags=re.M))" "$D4"
log "J1 joint fine-tune 시작"
python scripts/49_d3_joint.py --config configs/retrain/j1_joint.yaml --no-lock > $L/j1_joint.log 2>&1
rc=$?; log "J1 종료: $rc"; need_ok $rc J1
J1=$(ls -t outputs/checkpoints/retrain/j1_joint/*.pt 2>/dev/null | head -1)
need_ck "$J1"; log "J1 checkpoint = $J1"

# --- 7) A/B 추론 보정 -- val 에서만 (test 는 한 번만 쓴다) -------------------
log "A/B 추론 보정 (val) 시작"
python scripts/59_inference_tune.py --ckpt "$J1" --n-subj 12 > $L/b_tune.log 2>&1
rc=$?; log "A/B 종료: $rc"; need_ok $rc A/B

touch $L/PIPELINE_DONE
log "1~7 완료. test 는 수동으로 1회 실행한다 (49_eval_frozen.py --subjects outputs/splits/test.txt --by-count --resid-alloc)"
