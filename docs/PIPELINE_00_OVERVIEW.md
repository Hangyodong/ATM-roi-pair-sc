# 파이프라인 개요 — T1 단독 whole-brain TRK / SC 생성

작성 기준: 2026-09-04. 체크포인트 `outputs/checkpoints/route2/s5_joint/seg_full_step4000.pt`.

---

## 1. 목표

T1 강조 MRI **한 장만** 입력받아

1. whole-brain tractogram (streamline 집합)
2. 그로부터 추출한 82×82 structural connectivity (SC) 행렬

을 생성한다. DWI 는 학습 시 정답(GT) 을 만드는 데만 쓰고, 추론에는 쓰지 않는다.

## 2. 데이터

| 항목 | 값 |
|---|---|
| 코호트 | PPMI, T1 + DWI tractogram 이 모두 있는 **206명** |
| 분할 | train **144** / val **31** / test **31** |
| GT tractogram | subject 당 정확히 **1,000,000** streamline (DSI Studio `.tt.gz`) |
| 아틀라스 | Desikan + PD25, MNI152NLin6, 2mm, **82 ROI** (피질 66 + 피질하 16) |
| ROI 쌍 | 82×81/2 = **3,321** |

블록 구분: ctx-ctx **2,145** / ctx-sub **1,056** / sub-sub **120** 쌍.

## 3. 전체 흐름

```
[전처리]
  T1 (native)  ──ANTs 정합──▶  W 격자 193×229×193 @1mm  ──정규화──▶  모델 입력
  T1           ──ANTs Atropos──▶  WM 확률 맵                        (2번째 채널, 도입 중)
  .tt.gz       ──trans_to_mni──▶  streamline [N,128,3] mm
  streamline + 아틀라스 ──▶ sc_end / sc_pass / len_pass  (GT)
  streamline ──ROI 쌍별 분류(cap 256)──▶ bundles.npz
  streamline ──SC edge 단위 분해──▶ edge_segments.npz

[학습]
  T1 ──UNet──▶ anatomy a[1,512]
  (i,j) ──Emb──▶ pair_vec[64] ──Proj──▶ cond = gain·a + Proj + mode_emb
  GT streamline ──ConvVAE 인코더(cond)──▶ (mu, logvar) ──▶ z[64]
  z + cond ──ConvVAE 디코더──▶ streamline [N,128,3] mm
  + heads: weight / edge / count(pass) / count_end(끝점)

[추론]  ※ GT 를 전혀 쓰지 않는다
  T1 ──▶ a ──edge head──▶ 생성할 ROI 쌍
       ──배분(train 템플릿)──▶ 쌍마다 몇 가닥
       ──latent(train bank KDE)──▶ z
       ──디코더──▶ streamline ──아틀라스──▶ SC
```

## 4. 현재 성능 (test 31명, T1 만 입력)

| 구성 | SC 상관 r | CCC | 길이 RMSE | 잔차 r |
|---|---|---|---|---|
| 학습된 그대로 (prior + 균등 배분) | 0.711 | 0.682 | 55.0 mm | −0.002 |
| **추론 개선판** (latent bank + 템플릿 배분) | **0.880** | 0.737 | 41.3 mm | +0.026 |
| 참고: 그룹 평균 템플릿 (T1 을 안 봄) | **0.944** | 0.936 | — | 0 (정의상) |

- **재학습 없이 추론 방식만 바꿔 0.711 → 0.880.** (`docs/PIPELINE_04_INFERENCE.md`)
- **그룹 템플릿을 넘지 못한다.** 개인차 재현이 미해결 과제. (`docs/PIPELINE_06_FINDINGS.md`)

## 5. 문서 지도

| 파일 | 내용 |
|---|---|
| `PIPELINE_00_OVERVIEW.md` | 이 문서 |
| `PIPELINE_01_PREPROCESSING.md` | T1 정합·정규화, streamline 공간, SC 정의, bundle/segment, WM 분할 |
| `PIPELINE_02_MODEL.md` | 모델 구조, anatomy/streamline feature 추출, 조건화, head |
| `PIPELINE_03_TRAINING.md` | 손실 함수 전체, 5단계 phase, 균형 샘플링, GESTA |
| `PIPELINE_04_INFERENCE.md` | pair 선택, 배분, latent bank, SC 추출 |
| `PIPELINE_05_EVALUATION.md` | 평가 지표 정의와 전체 결과표 |
| `PIPELINE_06_FINDINGS.md` | 진단으로 밝혀진 결함, 폐기된 가설, 미해결 문제 |
| `PIPELINE_07_PROTOCOL_CHANGE.md` | 전처리 프로토콜 전환 (rigid / [0,1] / WM) 기록과 근거 |

## 6. 코드 지도

```
src/atm_sc/
  data/       prepare_t1.py  tt_io.py  dataset.py  roi_groups.py
              edge_segments.py  balanced_pair_sampler.py  segment_sampler.py
              bundle_statistics.py  wm_segment.py
  models/     roi_atm.py  atm_adapter.py  roi_pair_embedding.py
              edge_head.py  edge_count_head.py  streamline_weight_head.py
              endpoint_assigner.py
  losses/     geometry.py  endpoint.py  route.py  sc_corr.py  sc_magnitude.py
              edge.py  edge_count.py  tract_length.py  metrics.py
  training/   trainer.py  run.py
  inference/  generate_sc.py  latent_bank.py
  generative/ latent_sampler.py  bundle_augmenter.py
  filtering/  t1_streamline_filter.py  qc_thresholds.py
  evaluation/ balance_metrics.py

scripts/  01 좌표계 QC · 02 ROI 쌍 할당 · 03 bundle · 12 전처리 배치
          19 학습 드라이버 · 20 모니터 · 22 synthetic 캐시 · 24 visitation
          26 edge segment · 27 QC 임계값 · 29 최종 평가
          32 bank 추론 검증 · 33/35 SC 도표 · 34 추론 사전정보 생성
          36 WM 맵 · 37 rigid 정합
```

## 7. 전처리 프로토콜 변경 (진행 중)

기존 SyN(비선형) 정합에서 **rigid 정합 + [0,1] 정규화 + WM 채널**로 전환 중이다.
자세한 내용과 근거는 `PIPELINE_07_PROTOCOL_CHANGE.md` 참조. **입력이 바뀌므로 기존
체크포인트는 무효이고 재학습이 필요하다.**
