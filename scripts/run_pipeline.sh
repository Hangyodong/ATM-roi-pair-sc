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

# --- 1) 인코더: E8(stage3) 확정. E9(full) 은 드롭했다 ---------------------------
# 근거: 이 코드베이스의 unet_level=full 은 t1_encoder 23.7M 으로 stage3(22.4M) 대비
# +1.3M(5.9%) 뿐이고, 그 대가로 20.8GB 를 써서 A10 에서 OOM 난다 (19:08, 19:2x 두 번).
log "E 축: E8(stage3) 확정, E9(full) 드롭"

WIN=outputs/checkpoints/retrain/e4_unet_D_local_t1fp0.1/a1_resid_step3000.pt
log "E 축 승자(확정) = $WIN"

# --- 3) prior 재학습 (pair 별 저랭크, E8 위에서) -----------------------------
log "D-f (subject 조건부 prior) 시작"
python scripts/49_d3_joint.py --config configs/retrain/df_prior.yaml --no-lock > $L/df_prior.log 2>&1
rc=$?; log "D-f 종료: $rc"; need_ok $rc D-f
DF=$(ls -t outputs/checkpoints/retrain/df_prior/*.pt 2>/dev/null | head -1)
need_ck "$DF"; log "D-f checkpoint = $DF"

# --- 4) edge_head / count_head_end 재학습 ------------------------------------
rm -f $L/a3_aux.json   # 옛 결과가 남아 있으면 실패해도 그 경로를 집는다
log "A3 (edge_head + count_head_end) 시작"
python scripts/61_a3_aux.py --config configs/retrain/a3_aux.yaml --resume "${DF:-$WIN}" > $L/a3_aux.log 2>&1
rc=$?; log "A3 종료: $rc"; need_ok $rc A3
A3=$(python -c "import json;print(json.load(open('outputs/eval/a3_aux.json'))['checkpoint'])" 2>/dev/null)
need_ck "$A3"; log "A3 checkpoint = $A3"

# --- 5) 디코더 재학습 (좌표 상자 / 앵커) -------------------------------------
log "D-a/b/c 디코더+앵커 시작"
CK5="${A3:-${DF:-$WIN}}"
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
