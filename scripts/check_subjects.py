#!/usr/bin/env python
"""학습 대상 subject = .mat(GT SC) ∩ tracto 폴더(T1+.tt.gz). 겹치지 않는 쪽을 보고한다."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from atm_sc.data.paths import subject_overlap_report, subjects   # noqa: E402

r = subject_overlap_report()
print(f".mat (GT SC 있음)          : {r['mat']}")
print(f"tracto 폴더 (T1 + tt.gz)   : {r['folder']}")
print(f"교집합 = 학습/평가 대상    : {r['both']}")
print(f".mat 에만 있음 (tracto zip 에 없음 -> 원본에서 재추출 필요) : {len(r['mat_only'])}  {r['mat_only']}")
print(f"폴더에만 있음 (SC 없음 -> 제외)       : {len(r['folder_only'])}  {r['folder_only']}")
assert subjects() == sorted(set(subjects()))
assert r["both"] == len(subjects())
