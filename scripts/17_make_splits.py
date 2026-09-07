#!/usr/bin/env python
"""subject-level train/val/test split (최종 전략 §12). group(PD/HC) x batch2(scanner_proto) 로 stratify.
대상 = .mat ∩ tracto 폴더 (206명). 고정 seed. outputs/splits/{train,val,test}.txt 와 split.csv."""
import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from atm_sc.data.paths import subject_meta, subjects        # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--val", type=float, default=0.15); ap.add_argument("--test", type=float, default=0.15)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
meta = subject_meta()
subs = subjects()
rng = np.random.default_rng(a.seed)
strata = defaultdict(list)
for s in subs:
    strata[(meta[s]["group"], meta[s]["batch2"])].append(s)
split = {}
for key, ss in sorted(strata.items()):
    ss = list(ss); rng.shuffle(ss)
    n = len(ss); nt = int(round(n * a.test)); nv = int(round(n * a.val))
    for i, s in enumerate(ss):
        split[s] = "test" if i < nt else ("val" if i < nt + nv else "train")
out = ROOT / "outputs" / "splits"; out.mkdir(parents=True, exist_ok=True)
for part in ("train", "val", "test"):
    (out / f"{part}.txt").write_text("\n".join(sorted(s for s in subs if split[s] == part)) + "\n")
with open(out / "split.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["subject", "split", "group", "batch2", "proto"])
    for s in subs:
        w.writerow([s, split[s], meta[s]["group"], meta[s]["batch2"], meta[s]["proto"]])
print(f"{len(subs)} subjects -> " + ", ".join(f"{p}={sum(v == p for v in split.values())}" for p in ("train", "val", "test")))
for part in ("train", "val", "test"):
    c = Counter((meta[s]["group"], meta[s]["batch2"]) for s in subs if split[s] == part)
    print(f"  {part:5s}: " + "  ".join(f"{g}/{b}:{n}" for (g, b), n in sorted(c.items())))
assert not (set(open(out / 'train.txt').read().split()) & set(open(out / 'test.txt').read().split()))
print("->", out)
