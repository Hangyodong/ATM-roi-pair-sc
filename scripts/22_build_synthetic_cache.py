#!/usr/bin/env python
"""GESTA 전략 Phase 3–4: TRAIN subject 의 pooled latent bank + under-represented bundle synthetic cache.

  python scripts/22_build_synthetic_cache.py --ckpt outputs/checkpoints/phase2_geometry/geometry_step3000.pt --bank --augment
  python scripts/22_build_synthetic_cache.py --ckpt ... --augment --subjects sub-100001      # 1명 smoke

GPU 필요 (ES/DS). GT (assignments/bundles/.mat) 는 읽기만 한다 — 실행 전후 해시로 확인 (§79 SC target invariance).
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from atm_sc.data.dataset import ROIPairSubject                       # noqa: E402
from atm_sc.data.paths import ATLAS, CACHE                            # noqa: E402
from atm_sc.filtering.t1_streamline_filter import FilterConfig        # noqa: E402
from atm_sc.generative.bundle_augmenter import (AugmentConfig, LatentBank, augment_subject,   # noqa: E402
                                                build_latent_bank)
from atm_sc.models.roi_atm import ROIPairATM, TEMPLATE_MASK           # noqa: E402
from atm_sc.training.run import anatomy_feature                       # noqa: E402


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def main(a):
    import nibabel as nib
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sd = torch.load(ROOT / a.ckpt, map_location=dev, weights_only=False)
    lv = sd.get("unet_level", "none")
    model = ROIPairATM(n_roi=82, trainable="full" if lv == "full" else "vae", unet_level=lv, device=dev)
    model.load_checkpoint(sd["model"]); model.eval()
    subs = [l.strip() for l in (ROOT / a.subjects).read_text().splitlines() if l.strip()] if Path(ROOT / a.subjects).exists() else a.subjects.split(",")
    if a.limit:
        subs = subs[: a.limit]
    train = [l.strip() for l in (ROOT / a.train_subjects).read_text().splitlines() if l.strip()]
    assert set(subs) <= set(train), "augmentation seed 는 TRAIN subject 만 (§20 leakage 방지)"
    cfg = AugmentConfig(min_seed_count=a.min_seed, max_synthetic_ratio=a.max_ratio, method=a.method,
                        bw_factor=a.bw_factor, seed=a.seed, filter=FilterConfig())
    cfg.balance.alpha, cfg.balance.b_base, cfg.balance.b_min, cfg.balance.b_max = a.alpha, a.b_base, a.b_min, a.b_max
    cfg.bank_per_subject = a.bank_per_subject
    cfg.over_generate = a.over_generate
    from atm_sc.training.run import t1_input               # noqa: E402

    def anat(s):
        return anatomy_feature(model, s, a.init_bundle) if lv == "none" else model.atm.encode_anatomy(t1_input(model, s))
    out_root = ROOT / a.out
    bank_p = out_root / "latent_bank.npz"
    if a.bank:
        build_latent_bank(model, train[: a.bank_limit] if a.bank_limit else train, anat, bank_p, cfg)
    bank = LatentBank.load(bank_p) if bank_p.exists() else None
    if a.augment:
        assert bank is not None or a.no_bank, "latent bank 없음: --bank 먼저 (또는 --no-bank)"
        img = nib.load(ATLAS); atlas = np.asanyarray(img.dataobj).astype(np.int16)
        mimg = nib.load(TEMPLATE_MASK); mask = np.asanyarray(mimg.dataobj) > 0
        summaries = []
        for i, s in enumerate(subs):
            subj = ROIPairSubject(s)
            gt_files = [subj.dir / "assignments.npz", subj.dir / "bundles.npz"]
            h0 = [md5(p) for p in gt_files]
            summ = augment_subject(model, subj, anat(s), bank, cfg, atlas, img.affine, mask, mimg.affine, out_root / s)
            assert [md5(p) for p in gt_files] == h0, "GT 파일이 바뀜 (§79 위반)"
            summaries.append({k: v for k, v in summ.items() if k != "pairs"})
            print(f"[{i + 1}/{len(subs)}] done", flush=True)
        tot = {"n_subjects": len(summaries), "n_synthetic": int(sum(x["n_synthetic"] for x in summaries)),
               "n_real": int(sum(x["n_real"] for x in summaries)),
               "fill_rate_mean": float(np.mean([x["fill_rate"] for x in summaries])),
               "by_source": {k: int(sum(x["by_source"][k] for x in summaries)) for k in summaries[0]["by_source"]},
               "by_size_bin": {k: int(sum(x["by_size_bin"][k] for x in summaries)) for k in summaries[0]["by_size_bin"]},
               "by_block": {k: int(sum(x["by_block"][k] for x in summaries)) for k in summaries[0]["by_block"]},
               "mean_sampler_acceptance": float(np.mean([x["mean_sampler_acceptance"] for x in summaries if x["mean_sampler_acceptance"] is not None])),
               "mean_filter_pass": float(np.mean([x["mean_filter_pass"] for x in summaries if x["mean_filter_pass"] is not None])),
               "time": time.strftime("%F %T"), "ckpt": a.ckpt, "cfg": json.loads(json.dumps(cfg.__dict__, default=str))}
        (out_root / "augment_summary.json").write_text(json.dumps(tot, indent=1, ensure_ascii=False))
        print(json.dumps({k: v for k, v in tot.items() if k != "cfg"}, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--subjects", default="outputs/splits/train.txt", help="파일 경로 또는 콤마 목록")
    ap.add_argument("--train-subjects", default="outputs/splits/train.txt")
    ap.add_argument("--out", default="outputs/synthetic")
    ap.add_argument("--bank", action="store_true"); ap.add_argument("--augment", action="store_true"); ap.add_argument("--no-bank", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--init-bundle", default="AF_L")
    ap.add_argument("--method", default="kde", choices=["kde", "gaussian", "rejection"])
    ap.add_argument("--bw-factor", type=float, default=1.0)
    ap.add_argument("--min-seed", type=int, default=20); ap.add_argument("--max-ratio", type=float, default=4.0)
    ap.add_argument("--alpha", type=float, default=0.5); ap.add_argument("--b-base", type=float, default=100.0)
    ap.add_argument("--b-min", type=float, default=16.0); ap.add_argument("--b-max", type=float, default=256.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bank-per-subject", type=int, default=32)
    ap.add_argument("--over-generate", type=float, default=3.0, help="filter 탈락 대비 배수 (실측 통과율 0.115)")
    ap.add_argument("--bank-limit", type=int, default=0, help="bank 을 만들 train subject 수 제한 (smoke)")
    sys.exit(main(ap.parse_args()))
