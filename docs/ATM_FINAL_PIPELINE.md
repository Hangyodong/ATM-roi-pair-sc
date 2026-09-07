# ATM ROI-pair SC 파이프라인 — 최종 정리

**목표**: T1 한 장 → subject-specific whole-brain TRK → SC weight + SC tract-length
**데이터**: PPMI 206명 (GT SC `.mat` ∩ tractogram 폴더), subject 단위 split 144 / 31 / 31
**작성**: 2026-09-03 21:00 · 설계 근거 `docs/ATM_*_STRATEGY.md` 5종, 구현 이력 `ATM_FINAL_FINETUNING_REPORT.md`

---

## 0. 원본 ATM과의 차이

원본 ATM은 "이 T1을 보고 활꼴다발 하나를 그려줘"에 답한다. 다발 30종마다 모델 파일이 따로 있고,
공개본에 학습 코드는 없으며 SC는 다루지 않는다.

이 프로젝트는 "이 T1을 보고 **ROI i와 j를 잇는 연결**을 그려줘"에 답하는 모델 하나로 바꾼다.

| 항목 | 원본 ATM | 이 프로젝트 |
|---|---|---|
| 모델 | 다발별 30개 | 공유 1개 + ROI 쌍 임베딩 (82 ROI → 3,321쌍) |
| 조건 | 어느 `.pth`를 쓰느냐 | anatomy 특징 + ROI 쌍 + 생성 단위 모드(전체/segment) |
| T1 인코더 | 동결 | 미세조정 |
| 생성 개수 | 다발당 3,000 고정 | **연결마다 예측 개수** (GT 분포 재현) |
| 학습 코드 | 없음 | 전부 신규 |
| 출력 | 다발 하나의 streamline | whole-brain TRK + SC weight + SC length |

원본에서 쓰는 것은 **디코더 구조와 사전학습 가중치**뿐이다. `stable/stable/` 원본은 md5 대조로 무수정 확인.

---

## 1. 데이터의 사실 (모든 설계의 근거, 전부 실측)

| 사실 | 값 | 확인 방법 |
|---|---|---|
| GT tractogram 크기 | subject당 **1,000,000 가닥** | `.tt.gz` 디코딩 |
| 두 ROI를 끝점으로 갖는 가닥 | 488,906 (49 %) | 끝점 atlas 조회 |
| GT SC 정의 | 한 가닥이 **지나간 모든 ROI 쌍**을 센다 | Case B r=**0.9986** vs 인접 전이만 r=0.781 |
| pass-SC 총합 | 7,673,899 (가닥당 평균 7.7쌍) | 위 계산 |
| 연결당 개수 | 1 ~ 10,150 (중앙값 29) | `sc_end` 분포 |
| SC 값 규모 | 평균 2,311, 표준편차 5,478, 최대 55,551 | `.mat` |
| **그룹 평균 기준선** | r 0.945, CCC 0.936, RMSE 1,844 | 학습셋 GT 평균으로 test 예측 |
| subject 간 SC 상관 | 0.905 | 쌍별 계산 |
| 균등 배분의 상관 상한 | **0.854** (GT 개수대로면 0.970) | GT 기하 고정, 배분만 변경 |
| 실제 streamline QC 분포 | 길이 19~241 mm, 최대 꺾임각 ≤44.8°, 뇌 안 ≥0.95 | TRAIN 60,000가닥 p1/p99 |

> **가장 중요한 두 가지**
> 1. 목표는 "상관 1, RMSE 0"이 아니라 **그룹 평균(r 0.945, RMSE 1,844)을 넘는 것**이다. 못 넘으면 T1이 기여한 게 없다.
> 2. 연결마다 같은 개수를 만들면 기하가 완벽해도 상관 상한이 0.854다. **개수 배분이 손실 설계보다 큰 지렛대다.**

---

## 2. 전체 흐름

```text
                          T1 (native)
                               │  ANTs SyN → MNI152NLin6 → 193×229×193
                               ▼
                 Trainable T1 Encoder (ATM UNet 인코더, warm-start, 23.7 M)
                               │  anatomy feature [1,512]
     ┌─────────────┬───────────┴─────────┬──────────────┬────────────┐
     ▼             ▼                     ▼              ▼            ▼
 ROI-pair 조건  Edge Head          Count Head(pass)  Count Head(end)  Weight Head
 Emb(i)+Emb(j)  연결 유무          SC 값 예측        생성 개수 예측
     │  + 모드(0 전체 / 1 segment)
     ▼
 공유 ATM Decoder ──► mode 0: full streamline    mode 1: SC edge segment
     │                        │                            │
     │              기하·끝점·경로·길이 감독          segment 복원·끝점 감독
     ▼                        ▼                            ▼
        미분 가능한 PASS-SC (지나간 모든 ROI 쌍, 개수 가중)
                               │
     ┌──────────┬──────────────┼─────────────┬──────────┬──────────┐
     ▼          ▼              ▼             ▼          ▼          ▼
  corr 전체  corr CC/CS/SS  전역 배율    절대 크기    RMSE      tract length
```

---

## 3. 전처리 (학습 전 1회)

| 산출물 | 스크립트 | 내용 | 실측 |
|---|---|---|---|
| `cache/<sub>_T1w_syn_W.npy` | `01`, `12` | T1 → SyN → NLin6 → 1 mm 격자. 정합 후 가닥 90 % 이상 뇌 안인지 assert | 2~3분/명, 206/206 |
| `assignments.npz` | `02` | GT 100만 가닥의 끝점 ROI·쌍, endpoint/pass SC·길이 | 95초/명, 206/206 |
| `bundles.npz` | `03` | 쌍당 최대 256개를 128점 재샘플 (**실제 개수는 `pair_count_full`에 보존**) | 25초/명, 206/206 |
| `cache/dist_maps.npy` | `05` | 82 ROI 거리맵 (soft 끝점·통과 확률) | 1회 |
| `visit.npz` | `24` | 가닥별 지나간 ROI + 쌍별 통과 비율 = 경로 손실의 정답 | 3초/명, 206/206 |
| `edge_segments.npz` | `26` | 가닥을 **모든 방문 ROI 쌍의 부분경로**로 분해, 32점, edge당 128개 상한 | 99초/명, 진행 중 |
| `stats/qc_thresholds.json` | `27` | 합성 QC 임계값을 TRAIN 실제 분포 분위수에서 산출 | 1회 |
| `synthetic/*` | `22` | 작은 bundle 보강용 합성 가닥 (§4) | bank 4분 + 6초/명 |
| `splits/*.txt` | `17` | subject 단위 층화 분할 | 144 / 31 / 31 |

**분해 규모**: 가닥 155,273개 → segment 1,817,947개(가닥당 11.7개), edge 2,529개.
상한 후 저장 203,364개(11 %). 학습셋 29.3 M segment, 5.6 GB.
상한은 저장량 절감이자 편향 장치다 — 2,425 edge 중 **1,243개가 상한에 걸려** 큰 연결만 잘린다.

> **혼동 금지** (GESTA QC §20): `sc_mat`(GT 전체 tractogram의 pass-SC) = **SC 학습 목표**.
> `pair_count_full` / `edge_count_full`(상한 적용된 캐시 개수) = **증강·노출 균형용**. 둘은 다른 값이다.

---

## 4. 합성 streamline 생성과 QC

### 4.1 생성

```text
real 가닥/segment → ATM 인코더 → latent seed → KDE 샘플링 → ATM 디코더 → QC → 학습 풀
```

| 항목 | 규칙 | 근거 |
|---|---|---|
| 대상 선정 | TRAIN 크기 분포 **하위 25 % 미만**인 연결만 | 큰 bundle은 증강 불필요 |
| 생성량 | `제곱근 노출 목표 − 실제 개수`, 단 **실제의 4배 이하** | 과대 증폭 방지 |
| seed | 20개 이상이면 자체 KDE, 미만이면 **다른 subject의 같은 연결 latent 은행**, 그것도 없으면 쌍 사전분포 | KDE 안정성 |
| 학습 가중치 | real 1.0, **synthetic 0.5** | GT와 같은 신뢰도로 쓰지 않음 |
| 혼합 | real ≥ 50 %, 부족분은 real 복원 추출 | 합성 지배 방지 |

실측: 합성 총 1,339,614개, 채움률 0.45. 출처는 잠재 은행 893,437 / 자체 KDE 445,347.
64차원에서 rejection sampling은 채택률이 1e-11이라 사용 불가 — 직접 KDE 샘플링을 기본값으로 하고,
저차원에서만 rejection을 검증했다(조용한 fallback 없이 assert).

### 4.2 QC — 임계값을 실제 분포에서 뽑는다

| 기준 | 임계값 | 출처 |
|---|---|---|
| 끝점 일치 | 조건된 ROI 쌍에 양 끝이 닿아야 함 | hard reject |
| 길이 | 19.3 ~ 241.5 mm | TRAIN p1/p99 |
| 최대 꺾임각 | ≤ 44.8° | TRAIN p99 |
| 총 회전량 | ≤ 1265° | TRAIN p99 |
| 직선비 | ≥ 0.15 | TRAIN p1 (loop 탐지) |
| 뇌 안 비율 | ≥ 0.95 | TRAIN p1 |
| 근접 중복 | 평균 거리 ≥ 0.1 mm | 1.0 mm면 GT의 절반이 탈락 |
| 좌표 | NaN/Inf 없음 | 필수 |

이 임계값으로 **real 자신의 통과율 0.947**. 실패 사유별 통과 수를 연결마다 기록한다.

> **발견**: 이전에 쓰던 꺾임각 60°는 실제 분포(p99 44.8°)보다 **느슨했는데도** 합성의 74 %가 탈락했다.
> 즉 임계값이 엄격한 게 아니라 **합성물이 실제 분포 밖**이다. 생성 품질 자체를 봐야 한다.

---

## 5. 학습 — `configs/pipeline_route.yaml` (5단계 15,000 step)

학습 대상은 매 단계 동일: T1 인코더 23.7 M + VAE 인코더 0.89 M + 디코더 0.63 M + heads 1.01 M = **26.2 M**.
학습률 1e-5 / 3e-5 / 3e-5 / 1e-4 (s5는 1/2~1/3). 산출물 `outputs/checkpoints/route2/`.

| # | 단계 | 새로 켜지는 것 | 하는 일 | step |
|---|---|---|---|---|
| s1 | `s1_route` | recon·KL·geom, 끝점, **경로**, **생성 길이**, edge, **corr 4항**, **segment 분기**, **개수 예측 2종**, **전역 배율** | 모양·끝점·중간 경로·길이를 맞추고, SC 패턴을 4갈래로 맞추며, SC edge 단위 부분경로와 연결별 개수를 배운다 | 3,000 |
| s2 | `s2_presence` | + presence | 피질하 연결의 **존재 여부**를 크기와 분리해 학습 | 2,000 |
| s3 | `s3_mag` | + 절대 크기(총합 정규화 해제) + **RMSE** | 배율이 맞은 뒤 칸별 절대값을 다듬는다 | 3,000 |
| s4 | `s4_length` | + tract-length | 연결별 평균 길이 행렬 | 3,000 |
| s5 | `s5_joint` | 전체, 작은 LR | 전체 손실 동시 미세조정 | 4,000 |

### 손실

```text
L = 1.0·recon + 0.1·KL + 1.0·geom                         # full streamline 기하
  + 1.0·endpoint + 1.0·route + 0.5·route_gen              # 끝점 · 중간 통과 ROI
  + 1.0·gen_length                                        # 생성 길이를 그 쌍의 GT 평균 길이로
  + 0.5·edge                                              # 연결 유무
  + 0.5·(corr_CC+corr_CS+corr_SS)/3 + 0.5·corr_all        # SC 패턴 4항 (log 도메인)
  + 1.0·scale                                             # 전역 배율 |log Σpred − log Σgt|
  + 1.0·count(pass) + 1.0·count(endpoint)                 # 연결별 개수 log-MAE, block 균형
  + 1.0·seg_recon + 0.1·seg_KL + 1.0·seg_geom + 0.5·seg_endpoint   # SC edge 단위 segment
  + 0.5·presence                                          # SUB-SUB 존재 여부        (s2~)
  + 0.5·mag(절대) + 0.2·rmse(GT std 무차원화)              #                          (s3~)
  + 0.2·length                                            #                          (s4~)
```

### 100만 가닥을 학습에 반영하는 법

한 step에 100만 가닥은 불가능하다(5분/step). 대신 **중요도 가중**을 쓴다.

```text
연결마다 8개만 생성 · 각 가닥에 w = N̂(예측 개수) / 8 을 곱한다
→ 학습 중 계산하는 SC 가 "GT 개수대로 다 만들었을 때"의 불편추정이 된다
→ 이 가중치가 개수 예측 head 를 통과하므로 SC 손실이 "몇 가닥이어야 하는가"를 가르친다
```

실측: 자유 가중치일 때 총합비 0.028 → 개수 가중으로 0.614 (학습 전인데도).

### 한 step에서 보는 데이터

| 경로 | 표집 | 개수 |
|---|---|---|
| 생성 | 모든 쌍 × 8 | 약 15,000 가닥 |
| 복원 | 쌍 128개 × 8 (크기 제곱근 비례, real ≥ 50 %) | 1,024 |
| segment | edge 48개 × 8 (크기 제곱근 비례) | 384 |
| 개수 | 전체 3,321쌍 | 행렬 2개 |

---

## 6. 편향 완화 장치

| 문제 | 장치 | 실측 |
|---|---|---|
| 가닥 많은 bundle이 gradient 독점 | 노출 목표 = 크기의 제곱근 (16~256) | 250:1 → 5.7:1 |
| 작은 bundle의 기하 다양성 부족 | GESTA 잠재 증강 (real ≥ 50 %, 합성 ≤ 4배, 가중치 0.5) | 채움률 0.45 |
| 큰 edge가 상관을 지배 | corr 4항 분리 + log 도메인 | 피질하를 따로 감시 |
| 피질하 연결의 63 %가 통과로만 생김 | 경로 손실 + presence + segment 분기 | 통과 전용 edge 직접 감독 |
| 절대 크기 미학습 | 배율 항 + 절대 크기 + RMSE + 개수 head | 배율 하나로 CCC 0.024 → 0.817 |
| bundle 단위 ≠ SC edge 단위 | 모든 방문 쌍 부분경로 분해 | 분해 SC vs GT 상관 0.946 |
| 균등 생성이 상관을 0.854로 묶음 | 개수 예측 + 중요도 가중 + 추론 시 개수대로 생성 | 상한 0.854 → 0.970 |
| 경로 손실이 길이를 늘림 | 생성 길이 손실 + 경로 양성 가중 5 → 2 | 323 mm 폭주를 검출해 도입 |
| QC 임계값이 임의 상수 | TRAIN 분포 분위수에서 산출 | real 통과율 0.947 |

---

## 7. 검증

- **단계마다 자동**: 미학습 val 3명. block(CC/CS/SS) × 강도(소/중/대) 9칸 층화 pair 정확도, 경로 F1,
  SC 상관·순위상관·CCC·**RMSE**·**총합비**, 피질하를 endpoint 있는 것 / 통과 전용으로 분리.
- **최종 1회 자동**: `29_final_evaluation.py` — test 31명, **입력은 T1뿐**. 마지막 단계 후 자동 실행.
  생성 SC / 개수 기반 SC / **그룹 평균 기준선** / **잔차 상관**(개인차를 읽었는가)을 함께 보고.

### 트랙토그래피 재현 평가

`29_final_evaluation.py --trk-eval` 이 생성 tractogram의 기하를 GT와 비교한다 (GESTA QC §12, §60).

| 지표 | 뜻 | 기준선 (sc_corr checkpoint, 연결당 4개 생성) |
|---|---|---|
| coverage | GT가 지나는 복셀 중 생성물이 덮은 비율 | 0.710 |
| overreach | GT에 없는데 생성물이 침범한 복셀 비율 | 0.664 |
| dice | 두 복셀 집합의 겹침 | 0.598 |
| 길이 오차 / KS | 길이 분포 차이 | 36.6 mm / 0.445 |
| valid / duplicate | 유효 비율 / 근접 중복 비율 | 0.989 / 0.000 |

연결 단위 비교(큰 연결 50개)도 함께 낸다. 단 생성 수가 GT bundle보다 적으면 coverage가 구조적으로
낮게 나오므로 `--by-count`(연결마다 예측 개수만큼 생성)와 함께 읽어야 한다.

### 판정 기준

| 기준 | 값 |
|---|---|
| 최소 통과선 | 상관 ≥ 0.945, RMSE ≤ 1,844 (그룹 평균과 동급) |
| 실질적 성공 | 상관 ≥ 0.96, RMSE ≤ 1,500, **잔차 상관 > 0** |
| 비현실적 | RMSE 500 (상관 0.996 필요) |

---

## 8. 최종 산출물

```bash
python scripts/29_final_evaluation.py \
  --ckpt outputs/checkpoints/route2/s5_joint/seg_full_step4000.pt \
  --by-count --total-streamlines 489000 --trk
```

T1 → anatomy → edge head(어디가 연결됐나) → count head(몇 가닥인가) → 그 수만큼 생성 →
`.trk` 저장 → 같은 pass 규칙으로 SC weight/length 계산 → GT와 비교. GT는 지표 계산에만 쓴다.

---

## 9. 운영

| 항목 | 명령 / 동작 |
|---|---|
| 실행 | `python scripts/19_train_pipeline.py --config configs/pipeline_route.yaml` |
| 모니터 | `python scripts/20_monitor.py --watch 60` (파이프라인 자동 선택) |
| 이어받기 | 200 step마다 `<phase>_latest.pt`(모델+optimizer+RNG) |
| 새 head 추가 | `load_checkpoint`가 없는 키만 초기값으로 채운다 |
| A10 → H100 | `qsub scripts/pbs_route.sh` → preempt 파일로 양보 → H100이 이어받음 → 종료 시 로컬 자동 재개 |
| 코드 변경 반영 | driver는 기동 시점 코드를 들고 있다. phase 경계에서 재시작 필요 |
| 테스트 | `CUDA_VISIBLE_DEVICES="" python -m pytest -q tests/` (114개) |

---

## 10. 열린 문제

1. **합성 품질** — 생성물의 꺾임각이 실제 분포 밖이라 QC 통과율이 0.12다. 채움률 0.45는 여기서 온다.
   기하 손실 강화 또는 생성 후 평활화를 검토해야 한다.
2. **개인차** — 그룹 평균 대비 잔차 상관이 0.011로 사실상 0이다. T1 인코더가 subject를 구별하는지
   (사전학습 특징 코사인 0.95)가 관건이며, 미세조정 후 재측정이 필요하다.
3. **구간 내 상관** — 같은 크기 등급 안에서 상관이 0.17이다. 전체 0.83은 등급 간 대비에서 나온다.
   개수 예측과 segment 학습이 이를 겨냥한다.
4. **길이** — 경로 손실 도입 시 생성 길이가 323 mm까지 늘었다. 길이 손실로 대응 중이며 추적이 필요하다.
