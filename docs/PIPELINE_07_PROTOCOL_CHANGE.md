# 전처리 프로토콜 전환 기록 (2026-09-04)

확정된 프로토콜:

1. T1 을 **MNI152 공간으로 rigid registration**
2. intensity 를 **[0,1]** 로 정규화
3. streamline 도 같은 MNI 공간으로 변환
4. T1 에서 **white-matter** 추출 (FreeSurfer → ANTs 대체)

**모델 입력이 바뀌므로 기존 체크포인트는 무효이고 재학습이 필요하다.**

---

## 1. 이전 상태와 무엇이 달라지는가

| # | 항목 | 이전 | 확정 프로토콜 | 상태 |
|---|---|---|---|---|
| 1 | T1 정합 | ANTs **SyN** (비선형) | ANTs **Rigid** | ✅ 206/206 완료 |
| 2 | intensity | 99.5 백분위를 0.6 에 맞춤 (**최대 1.24**) | 99.5 백분위를 1.0 에, 위를 clip → **[0,1]** | ✅ 구현 |
| 3 | streamline | `trans_to_mni` 로 MNI mm | 동일 | ✅ 이미 됨 |
| 4 | WM | 없음 | ANTs Atropos WM 확률 볼륨 | 🔄 206명 생성 중 |

## 2. ① Rigid 정합

`scripts/37_build_rigid_t1.py` — 206명, 6병렬, 약 52초/명.

```python
ants.registration(fixed=TEMPLATE, moving=T1, type_of_transform="Rigid")
→ resample_to_W(order=1) → [193,229,193] float32
→ outputs/cache/{sub}_T1w_rigid_W.npy
```

### 왜 바꿨나

SyN 은 각 뇌를 템플릿 모양으로 구부린다. 개인의 뇌 크기·모양·국소 부피가 **워프 필드로
빠져나가는데 그 필드를 저장하지 않았다.** 8명 실측:

| 정합 | subject 간 영상 상관 | 개인 변동계수 |
|---|---|---|
| SyN | 0.900 | 0.239 |
| **Rigid** | **0.731** | **0.453** (1.89배) |

Rigid 가 개인 해부 변동을 1.89배 유지한다.

### 반대 방향의 측정도 있다 (정직하게 기록)

같은 80명(train 60 / test 20)에서 T1 → SC 개인차를 ridge 로 예측해 비교하면:

| 입력 | PCA | test 잔차 r |
|---|---|---|
| SyN | 80 | **0.175** |
| Rigid | 40 | 0.104 |

**선형 probe 에서는 SyN 이 낫다.** 이유는 대응(correspondence)이다 — rigid 는 복셀 (i,j,k)
가 사람마다 다른 해부 위치라 선형 모델이 쓰지 못한다. Rigid 의 큰 변동 중 상당 부분은
정렬 불일치다.

다만 이 probe 는 **선형 모델**이고, 실제 인코더는 3D CNN 이라 정렬에 덜 민감할 수 있다.
프로토콜은 사용자 지시에 따라 **Rigid 로 확정**한다. 이 절충은 Methods 에 명시해야 한다.

### 알아둘 결과

streamline/아틀라스는 MNI(QSDR 비선형 정규화) 공간에 있고 rigid T1 은 복셀 단위로 그것과
정확히 대응하지 않는다. 인코더는 전역 특징 벡터 하나를 뽑으므로 복셀 대응이 필수는
아니지만, **T1 입력만 rigid 로 바꾸고 SC 계산·ROI 할당은 기존 좌표계를 유지**한다.
(아틀라스가 MNI 에 정의돼 있어 rigid 공간에서 ROI 할당을 다시 하면 부정확해진다.)

## 3. ② intensity [0,1]

`BundleNorm.normalize_t1(..., unit=True)`

```python
scale = percentile(vol[vol > 0], 99.5)
out   = clip(vol / scale, 0.0, 1.0)
assert 0.0 <= out.min() and out.max() <= 1.0
```

이전 방식은 99.5 백분위를 0.6 에 맞추는 것이라 상위 0.5 % 가 1 을 넘었다:

| subject | 이전 최대 | `unit=True` |
|---|---|---|
| sub-101070 | 0.890 | 1.000 |
| sub-100001 | **1.208** | 1.000 |
| sub-4030 | **1.237** | 1.000 |

백분위 기준을 유지하는 이유: PPMI T1 은 native max 가 878 ~ 203,163 으로 230배 차이나고,
단순 min-max 를 쓰면 밝은 이상치(지방·혈관) 하나가 전체 스케일을 지배한다.

## 4. ④ White matter — FreeSurfer 대체

### 왜 대체했나

| | FreeSurfer `recon-all` | **ANTs Atropos** |
|---|---|---|
| 시간 (1명) | 6~10시간 | **~35초** |
| 시간 (206명, 6병렬) | **10~14일** | **~20분** |
| 설치 | 안 되어 있음 (라이선스 필요) | 이미 있음 |
| 산출물 | 정점 메시 (표면) | WM 확률 볼륨 |
| 인코더 적합성 | 메시 → 볼륨 변환 필요 | 3D UNet 에 바로 |

인코더가 3D UNet 이라 표면 메시를 받아도 결국 볼륨으로 되돌려야 한다.

### 방법

```python
mask = brain_mask_W(erode_mm=4)
seg  = ants.atropos(a=T1_rigid, x=mask, i="kmeans[3]", m="[0.2,1x1x1]", c="[5,0]")
wm   = 확률맵 중 가중평균 강도가 가장 높은 클래스     # T1 에서 WM > GM > CSF
```

클래스 번호를 고정으로 가정하지 않고 **강도 순위**로 고른다 (k-means 초기화는 순서를 보장하지 않는다).

### 여기서 실제 결함을 하나 잡았다

첫 QC 그림에서 뇌실 주변에 **직사각형 블록**이 보였다. 원인 두 가지:

| 문제 | 결과 |
|---|---|
| 템플릿 뇌 마스크가 뇌실을 **구멍으로 남김** (10,714 복셀) | 침식 시 구멍이 **66,329 복셀**로 부풀어 뇌실 주변 백질이 통째로 잘림 |
| `np.ones((3,3,3))` 정육면체 구조요소 | 상자 모양 인공물 |

수정:

```python
m = binary_fill_holes(m)          # 먼저 메운다. 뇌실은 Atropos 가 CSF 로 분류하므로 안전
m = binary_erosion(m, ball(4))    # 구형 구조요소
assert not (binary_fill_holes(m) & ~m).any(), "마스크에 내부 구멍이 남아 있다"
```

수정 후: 마스크 1,528,955 복셀, WM 42.1 % (정상 40~45 %), 인공물 없음.
QC 그림 `outputs/figures/wm_qc_rigid.png` 에서 뇌량·내포·소뇌 백질이 잡히고 뇌실·피질은 제외됨을 확인.

### 품질 관문

```python
assert 0.20 < WM비율 < 0.70              # 뇌 마스크 안에서
assert isfinite(wm).all() and wm.max() > 0.9
assert m.sum() > 800_000                 # 마스크가 비지 않았는가
```

WM 확률이 마스크 밖·구멍 안에서 0 인 것도 확인했다 (누출 없음).

## 5. 모델 입력 변경 (예정)

현재 인코더 입력은 `[1, 1, 193, 229, 193]` 단일 채널이다. WM 을 2번째 채널로 넣는다:

```
[1, 2, 193, 229, 193]   채널 0 = rigid T1, [0,1] 정규화
                        채널 1 = WM 확률, 0~1
```

UNet 첫 conv 를 `in_channels=1 → 2` 로 확장하고 **새 채널 가중치를 0 으로 초기화**한다.
ATM 사전학습 가중치를 보존한 채 WM 경로만 새로 학습된다.

**주의**: WM 맵은 T1 에서 결정론적으로 계산된 것이라 정보가 새로 생기지 않는다. 학습을
쉽게 만드는 유도 편향으로서의 가치이고, anatomy 경로의 크기·구조 불균형
(`PIPELINE_02_MODEL.md` §4, `PIPELINE_06_FINDINGS.md` §4)을 같이 고치지 않으면
여전히 무시될 수 있다.

## 6. 재실행 순서

```bash
python scripts/37_build_rigid_t1.py --workers 6     # ✅ 206/206 완료
python scripts/36_build_wm_maps.py  --workers 6     # 🔄 진행 중 (~20분)
# 이후: 모델 2채널화 → 5단계 phase 재학습 → scripts/34 → scripts/29
```

## 7. 산출물

| 파일 | 개수 | 내용 |
|---|---|---|
| `outputs/cache/{sub}_T1w_rigid_W.npy` | 206 | rigid 정합 T1, 정규화 전 |
| `outputs/cache/{sub}_WM_W.npy` | 206 (생성 중) | WM 확률 0~1 |
| `outputs/figures/wm_qc_rigid.png` | 1 | WM 분할 QC |
