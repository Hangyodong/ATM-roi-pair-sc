#!/usr/bin/env python
"""subject 간 anatomy feature 분산 점검.

pretrained UNet 의 feature 는 |a|~0.09 로 매우 작다. T1 -> SC 학습이 성립하려면 subject 마다
feature 가 달라야 한다. T1 W 캐시가 있는 subject 전부에 대해 feature 를 만들고
  - between-subject std (채널 평균) vs bias-only feature(T1=0) 로부터의 거리
  - subject 쌍 코사인 유사도
를 보고한다.
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from atm_sc.data.paths import CACHE                  # noqa: E402
from atm_sc.models.roi_atm import ROIPairATM         # noqa: E402
from atm_sc.training.run import anatomy_feature      # noqa: E402

import argparse
ap = argparse.ArgumentParser(); ap.add_argument("--mode", default="syn", choices=["syn", "rigid"]); a = ap.parse_args()
subs = sorted(p.name.replace(f"_T1w_{a.mode}_W.npy", "") for p in CACHE.glob(f"sub-*_T1w_{a.mode}_W.npy"))
assert subs, "T1 W 캐시 없음"
dev = "cuda" if torch.cuda.is_available() else "cpu"
m = ROIPairATM(n_roi=82, device=dev)
def feat(s):
    t1 = np.load(CACHE / f"{s}_T1w_{a.mode}_W.npy")
    x = torch.tensor(m.norm.normalize_t1(t1).reshape(1, 1, *t1.shape), dtype=torch.float32)
    return m.encode_anatomy(x)
F = torch.cat([feat(s) for s in subs]).cpu().numpy()      # [S,512]
print(f"mode={a.mode}")
a0 = m.encode_anatomy(torch.zeros(1, 1, 193, 229, 193)).cpu().numpy()          # bias-only
print(f"subjects: {len(subs)}  feature dim {F.shape[1]}")
print(f"|a| mean {np.linalg.norm(F,axis=1).mean():.4f} | |a - a0| mean {np.linalg.norm(F-a0,axis=1).mean():.4f} | |a0| {np.linalg.norm(a0):.4f}")
if len(subs) > 1:
    D = F - a0
    print(f"between-subject std (채널 평균) {F.std(0).mean():.5f}  vs 채널 |mean| {np.abs(F.mean(0)).mean():.5f}")
    Dn = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-12)
    cos = Dn @ Dn.T; iu = np.triu_indices(len(subs), 1)
    print(f"해부 성분(a-a0) 쌍별 cosine: mean {cos[iu].mean():.3f}  min {cos[iu].min():.3f}  max {cos[iu].max():.3f}")
    print("  (cosine ~1 이면 subject 구분 정보가 거의 없다는 뜻)")
    pd = np.linalg.norm(D[:, None] - D[None], axis=-1)
    print(f"subject 쌍 L2 거리: mean {pd[iu].mean():.4f}  vs 해부 성분 크기 {np.linalg.norm(D,axis=1).mean():.4f}")
for s, f in zip(subs, F):
    print(f"  {s}: |a|={np.linalg.norm(f):.4f}")
