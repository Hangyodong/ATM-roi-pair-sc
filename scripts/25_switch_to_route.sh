#!/usr/bin/env bash
# baseline 파이프라인을 "지금 돌고 있는 phase 까지만" 실행하고 ROUTE(+GESTA) 파이프라인으로 전환한다.
#   nohup setsid bash scripts/25_switch_to_route.sh > outputs/checkpoints/route/switch.log 2>&1 &
# 1) 현재 phase 의 최종 checkpoint + val 기록이 나올 때까지 대기
# 2) baseline driver 종료 (다음 phase 로 넘어가지 못하게)
# 3) GESTA latent bank + synthetic cache 생성 (그 checkpoint 로)
# 4) configs/pipeline_route.yaml (s1~s5) 실행
set -u
cd "$(dirname "$0")/.."
mkdir -p outputs/checkpoints/route
CKPT=${CKPT:-outputs/checkpoints/phase6_sc_corr/sc_corr_step3000.pt}
PHASE_NAME=${PHASE_NAME:-sc_corr}
SKIP_GESTA=${SKIP_GESTA:-0}

echo "[switch] $(date) 대기: $CKPT + val 기록($PHASE_NAME)"
while true; do
  if [ -f "$CKPT" ] && grep -q "\"phase\": \"$PHASE_NAME\"" outputs/checkpoints/val_metrics.jsonl 2>/dev/null; then break; fi
  sleep 30
done
echo "[switch] $(date) phase 완료 확인"

PID=$(pgrep -f "[1]9_train_pipeline.py --config configs/pipeline.yaml" | head -1)
if [ -n "$PID" ]; then
  echo "[switch] baseline driver $PID 종료"
  kill "$PID"
  for _ in $(seq 60); do kill -0 "$PID" 2>/dev/null || break; sleep 2; done
  kill -9 "$PID" 2>/dev/null
fi
rm -f outputs/checkpoints/pipeline.lock
sleep 20                                        # GPU 메모리 반환 대기

if [ "$SKIP_GESTA" != "1" ]; then
  if [ ! -f outputs/synthetic/latent_bank.npz ]; then
    echo "[switch] $(date) latent bank 생성"
    python scripts/22_build_synthetic_cache.py --ckpt "$CKPT" --bank || exit 1
  fi
  if [ ! -f outputs/synthetic/augment_summary.json ]; then
    echo "[switch] $(date) synthetic cache 생성 (train 144명)"
    python scripts/22_build_synthetic_cache.py --ckpt "$CKPT" --augment || exit 1
  fi
fi

echo "[switch] $(date) ROUTE 파이프라인 시작"
exec python scripts/19_train_pipeline.py --config configs/pipeline_route.yaml
