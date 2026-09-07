#!/usr/bin/env python
"""추론에 쓸 두 가지를 train subject 로만 만들어 저장한다 (test 는 일절 보지 않는다).

  1) 배분 템플릿  : train 평균 sc_end / sc_pass   -> outputs/inference/template.npz
  2) latent bank : train streamline 의 latent    -> outputs/inference/latent_bank.npz

왜: 학습된 count_head_end 는 로그 과분산으로 배분을 망치고(균등보다 나쁨), 사전분포
N(mu_pair, I) 는 실제 다발이 없는 자리를 그린다. 둘 다 train 통계로 대체하면 재학습 없이
SC 상관이 0.68 -> 0.89 로 올라간다 (scripts/32_verify_bank_inference.py 로 검증).

  python scripts/34_build_inference_prior.py --n-bank-subj 12
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from atm_sc.inference.latent_bank import build_bank                # noqa: E402
from atm_sc.models.roi_atm import ROIPairATM                       # noqa: E402

OUT = ROOT / "outputs/inference"


def main(a):
    train = [l.strip() for l in open(ROOT / "outputs/splits/train.txt") if l.strip()]
    test = set(l.strip() for l in open(ROOT / "outputs/splits/test.txt") if l.strip())
    assert not (set(train) & test), "train/test 가 겹친다"
    OUT.mkdir(parents=True, exist_ok=True)

    # --- 1. 배분 템플릿 -------------------------------------------------------
    # edge_prob 는 "그 pair 에 edge 가 있던 subject 비율" 이다. 학습의 edge 양성 정의와
    # 같은 기준(sc_end > 0)을 써야 edge head 인수분해가 의미를 갖는다 (재학습 설계 §2 ③).
    te = tp = ep = None
    for s in train:
        z = np.load(ROOT / f"outputs/roi_pairs/{s}/assignments.npz")
        te = z["sc_end"].astype(np.float64) if te is None else te + z["sc_end"]
        tp = z["sc_pass"].astype(np.float64) if tp is None else tp + z["sc_pass"]
        e = (z["sc_end"] > 0).astype(np.float64)
        ep = e if ep is None else ep + e
    te /= len(train); tp /= len(train); ep /= len(train)
    iu = np.triu_indices(len(te), 1)
    assert te.sum() > 0 and np.allclose(te, te.T), "sc_end 템플릿이 비었거나 비대칭"
    assert np.allclose(ep, ep.T) and 0 <= ep.min() and ep.max() <= 1, "edge_prob 가 확률이 아니다"
    np.savez_compressed(OUT / "template.npz", sc_end=te, sc_pass=tp, edge_prob=ep,
                        n_subjects=len(train), subjects=np.array(train))
    print(f"템플릿 저장: sc_end 비영 {int((te[iu] > 0).sum())} pair, "
          f">=1 인 pair {int((te[iu] >= 1).sum())}, 총합 {te[iu].sum():,.0f} 가닥, "
          f"edge_prob 평균 {ep[iu].mean():.3f}")

    # --- 2. latent bank -------------------------------------------------------
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sd = torch.load(ROOT / a.ckpt, map_location=dev, weights_only=False)
    lv = sd.get("unet_level", "full")
    m = ROIPairATM(n_roi=len(te), trainable="full" if lv == "full" else "vae", unet_level=lv, device=dev)
    m.load_checkpoint(sd["model"]); m.eval()
    bank = build_bank(m, train[:a.n_bank_subj],
                      lambda s: ROOT / f"outputs/roi_pairs/{s}/bundles.npz",
                      n_per_pair=a.bank_per_pair, seed=a.seed)
    p = bank.save(OUT / "latent_bank")
    cov = bank.n_pairs / max(int((te[iu] >= 1).sum()), 1)
    print(f"bank 저장: {p}  ({bank.n_pairs} pair, 생성 대상 pair 의 {cov:.1%} 덮음)")
    assert cov > 0.9, f"bank 가 생성 대상 pair 의 {cov:.1%} 밖에 못 덮는다 -- --n-bank-subj 를 늘려라"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/checkpoints/route2/s5_joint/seg_full_step4000.pt")
    ap.add_argument("--n-bank-subj", type=int, default=12)
    ap.add_argument("--bank-per-pair", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
