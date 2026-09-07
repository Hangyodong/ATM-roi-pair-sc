## W3-a 이후 (2026-09-07)
- [ ] 생성 기하(dice)가 valid_conn 과 같이 안 오른다: endpoint 는 맞는데 경로가 GT 번들에서 벗어난다.
      route/gen_length 가중치나 pair 단위 기하 손실을 직접 거는 쪽을 검토 (현재 dice 를 직접 겨냥하는 항이 없다).
- [ ] prior mu 가 posterior 평균에서 멀어지는 문제 (offset 3.41 -> 5.8). prior 적합항이 mu 를 못 따라간다.
      prior_mu 전용 LR 그룹(지금은 heads 와 공용)이나 posterior 쪽을 prior 로 당기는 대칭 항을 검토.
- [ ] `tests/test_sc_scale.py::test_count_weight_mode_gives_gt_scale` 는 W3-a 이전부터 실패 중 (난수 초기화 의존, 비율 10 vs 기준 4~6.5).

# TODO

## 즉시
- [x] **평가 규약 고정** (W3-b, 2026-09-07): `outputs/eval/w3b_protocol.json`. dice 는 규약 없이 인용 금지.
      `wb@N` 은 n_pred==n_gt==N, `pair@64` 는 pred=gtA=gtB=64. 모델·기준선 대응표는
      `outputs/eval/w3b_baselines_matched.json`, 재생성은 `python scripts/51_w3b_protocol_report.py`.
      **판정 유지: 모델은 cross_subject 기준선을 넘지 못한다** (wb@8000 0.546 vs 0.574, wb@20000 0.608 vs 0.629).
- [x] **개인차 축 종결** (W4, 2026-09-07, `scripts/52_prior_anatomy.py`): ROI 국소 조건화·경로 조건화를
      모두 프로브로 검정했다. 세 타깃(모양/세기/위치) x 세 위치(전역/끝점/경로) 중 유의한 칸은
      **전역 anatomy -> SC 0.101** 하나뿐이고 그건 머리 크기다. feature 는 subject 성분 64% 로 풍부한데
      타깃과 무관하다. **프로브 계열 실험은 여기서 멈춘다** (선형 readout 상한은 다 재봤다).
      PROJECT_STATUS 의 W4 절, `docs/PIPELINE_12_PROBLEM_SUMMARY.md` 참조.
- [ ] **[다음-A] 잔차 타깃 end-to-end 학습** — 개인차 축의 데이터 수준 가설을 검정하는 **유일하게 안 해본
      직접 실험**. count head 를 SC 절대값이 아니라 `SC - 그룹템플릿` 잔차에 학습시킨다.
      supervision 을 직접 걸고도 resid_r 이 0 이면 그때 "T1 에 정보가 없다"가 근거를 갖고, 0 이 아니면
      프로브 음성은 frozen 인코더 탓이었다는 뜻이다. 어느 쪽이든 새 정보를 얻는다.
      현재 `residual_r` 0.023 +- 0.162, 프로브 상한 0.101(머리 크기).
- [ ] **[다음-B] 그룹 기하 품질 3종** — 그룹 재현조차 cross-subject 기준선 미달이라 회수 여지가 크다.
      (템플릿 상수 r 0.945 vs 모델 0.788, 0/31)
      1. prior 온도 스윕 `z = mu + s·sigma·eps` (s 0.2~1.0). 학습 0. prior sigma 1.0 vs GT 조건 내 sd 0.18
      2. 가닥 수 보정: 예측 총합이 GT 의 8.9%. 전역 상수 k=11.2 로 CCC 0.090 -> 0.746
      3. 후처리 필터 (GM/WM 마스크 + minlength 20, 상류 ATM 의 tckedit 대응). overreach 1.407 vs GT 0.212
- [ ] 평가에서 그룹 정보 구성 제거: `scripts/29_final_evaluation.py` 의 `--use-bank` / 템플릿 배분은 보고에 쓰지 않는다. `generated` 단독 + 그룹 템플릿 기준선만.
- [x] subject 간 anatomy feature 분산 측정 — 완료 (2026-09-05): rigid+WM 재학습 후 ‖a‖ 0.785 인데 subject 간 코사인 0.9996. 크기는 해결, 분산은 아님
- [ ] `.mat` 에 있으나 tracto zip 에 없는 32 명 (`outputs/subjects_missing_in_tracto_zip.txt`) — 원본 머신에서 재추출할지 결정
- [ ] 206명 전처리 배치: 01(SyN, ~6min/명) → 02(~90s) → 03(~20s). background.

## GPU job
- [ ] interactive 세션 종료 후 `qstat -f 82650.KITSM02` 로 R 확인, `outputs/pbs/`·`outputs/checkpoints/phase2_geometry/log.jsonl` 초기 로그 검증
- [ ] CPU job 82756 (전처리 잔여) 완료 확인: `outputs/pbs/atm_sc_preproc.o82756`, `outputs/preprocess_logs/summary.csv`
- [ ] 로컬 파이프라인이 phase 를 넘어갈 때 val_metrics.jsonl 확인 (pair_acc, sc_pass_r_w 상승 여부)

## 학습
- [x] loss 스케일 정렬 (recon mm 단위, clip 50). 그룹별 gnorm 은 log.jsonl
- [ ] 길이 과대(96~108 vs 59 mm)·SC 절대 스케일(CCC 0.1) 개선: pair 길이 prior, 정규화 없는 magnitude 항
- [ ] Phase 2 baseline 수백 step → L_recon 수렴 확인 → Phase 3~8 순서대로
- [ ] inference 스크립트 (§22) + `.mat` pass-SC 대비 평가
- [ ] subject split: group × batch2 stratify

## 확인
- [ ] scipy 1.15.3 다운그레이드가 다른 프로젝트에 영향 없는지
- [ ] `ppmi_probe/` (5명 중복, 600MB) 삭제 여부
- [ ] phase 6~9 진행 후 block 별(ctx-ctx/ctx-sub/sub-sub) r_log 와 tier 별 pair_acc 가 함께 오르는지 확인. sub-sub 는 bundle 이 12~78 pair 뿐이라 통과 streamline 으로만 재현됨 — 안 오르면 block_weights 조정
- [ ] baseline 종료 후 scripts/23_after_baseline.sh 자동 실행 확인 (latent bank -> synthetic cache -> configs/pipeline_route.yaml 8 phase)
- [ ] ROUTE ablation: route 유무 x GESTA 유무로 pass-only SUB-SUB corr/recall 비교 (문서 §60-62)
- [ ] Weight Head shortcut 점검: SC corr 는 오르는데 route F1/endpoint acc 가 떨어지면 magnitude 가중치 하향 (§63)
- [ ] edge_segments.npz 206명 완료 확인 (scripts/26, ~2h) 후 edge 별 count 분포/불균형 통계
- [ ] Edge Count Head(§20-22) 도입 검토: (anatomy, ROI_i, ROI_j) -> log count. SC 절대 스케일(CCC 0.02) 직접 겨냥
- [ ] segment 생성 분기(dual representation)는 decoder 구조 변경이므로 route/GESTA 결과를 본 뒤 결정
