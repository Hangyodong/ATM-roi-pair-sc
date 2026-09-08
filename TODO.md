# TODO (2026-09-08 밤 기준)

규칙: 모든 선택은 val 31명. **test 31명은 이미 1회 소진** (2026-09-08 01:39, `docs/PIPELINE_13` M24).
새 판정이 필요하면 val 이나 새 split 으로만.

## 지금 무인으로 도는 것 (`scripts/82_auto_agenda.sh`)
- [ ] **A. trimming 판정** — `w_rl_val`(끔) vs `w_rl_val_trim`(백질 자르기만, 버리기 없음).
      `83_compare_runs.py` 가 부트스트랩 CI 로 판정한다. trimming 판에서 A_rowsum 38 -> 9.4
- [ ] **B. 백질 학습 기여 분리** — 이긴 설정을 J1(백질 학습 전) 에 걸어 W1 과 비교
- [ ] **C. alpha 선택 편향 제거** — alpha 를 train 20명에서 고르고 val 로 보고

## 확정된 것 (val 31명)
- [x] RL 역산 + 능선 목표 + 선형 진폭: resid_r 0.0016 -> **0.0813**, inter 0.914 -> **0.886**
      (GT 0.904), 절대 r 0.595 -> **0.810**. `docs/PIPELINE_13` M32
- [x] 능선 60 feature val 상한 **0.153** (count head 0.086)
- [x] 백질 점유 손실 W1: 점유 0.418 -> 0.473, 복원 4.412 -> 4.669 (게이트 4.6 근소 초과)
- [x] 새 추출 3종 + corridor. 새 다운로드 없이 전부 로컬 도구로

## 일반화 — 주장하려면 필요한 것
- [ ] **중첩 교차검증**. 오늘 val 로 고른 것: feature 블록, alpha, 표본 수, iters, damp, 스윕 팔,
      trimming 여부. val 은 이제 선택 집합이라 0.081/0.153 은 낙관적이다
- [ ] 외부 코호트 (HCP/ADNI 를 같은 전처리로). 단일 코호트 144명 학습이라 스캐너 의존이 걱정된다
- [ ] 가장 강한 feature 가 rigid 공간의 머리 크기·부피·거리다. `t1_scale` 은 코드가 이미
      "교란변수" 로 표시해둔 값이다

## 코드 (우선순위 순)
- [ ] W1 복원 4.669 > 게이트 4.6. 백질 가중치를 낮추거나 step 을 줄여 재학습할지 B 결과 보고 판단
- [ ] 신경망 head 의 tier1 feature 는 아직 9개. corridor/surf/tissue 를 넣으면 `tier1_w` 모양이
      바뀌어 head 3개 재학습 필요. 능선이 이미 그 정보를 쓰므로 급하지 않다
- [ ] 인코더 입력은 여전히 T1 + 옛 WM 2채널. 새 조직맵을 넣으면 인코더 전체 재학습
- [ ] 최적 step checkpoint 보존: 지금은 마지막 것만 남아 `max_steps` 가 곧 모델 선택
- [ ] `79_pick_trim.py` 목적함수 재검토 — 자르기와 버리기를 한 덩어리로 판정해 처음에 틀렸다

## 데이터 품질
- [ ] 재현되는 WM 비율 극단값 5명 (`outputs/eval/wm_qc.json`). T1 품질/정합 확인 필요
- [ ] 추적 파일에 남은 실제 subject ID: `docs/*.md` 의 sub-1135, sub-4030 (이미 push 됨).
      새로 만든 파일은 정리했다
- [ ] `.mat` 에 있으나 tracto zip 에 없는 32명, `ppmi_probe/` (5명 중복, 600MB) 삭제 여부
