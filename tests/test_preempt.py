"""A10 -> H100 인계: preempt 파일이 있으면 latest 를 저장하고 양보한다 (실제 subject, CPU)."""
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SUB = "sub-000001"
NEEDED = [ROOT / "outputs/roi_pairs" / SUB / "bundles.npz",
          ROOT / "outputs/cache" / f"{SUB}_anat_AF_L.npy",
          ROOT / "outputs/cache/dist_maps.npy"]


@pytest.mark.skipif(not all(p.exists() for p in NEEDED), reason="전처리 산출물 필요")
def test_preempt_saves_latest_and_raises(tmp_path):
    from atm_sc.training.run import Preempted, run
    from atm_sc.training.trainer import TrainConfig
    pre = tmp_path / "preempt"; pre.touch()
    out = tmp_path / "ck"
    cfg = TrainConfig(active={"recon"}, gt_pairs_per_step=2, n_gt_per_pair=2, seed=0)
    with pytest.raises(Preempted):
        run(phase="geometry", subjects=[SUB], max_steps=500, out_dir=out, cfg=cfg, device="cpu",
            trainable="vae", unet_level="none", log_every=1000, save_every=0, preempt_file=pre)
    ck = out / "geometry_latest.pt"
    assert ck.exists()
    sd = torch.load(ck, map_location="cpu", weights_only=False)
    assert sd["step"] == 10 and sd["phase"] == "geometry" and "optimizer" in sd   # 10 step 마다 확인

    pre.unlink()                                   # preempt 해제 후에는 끝까지 돈다
    ck2 = run(phase="geometry", subjects=[SUB], max_steps=12, out_dir=out, cfg=cfg, device="cpu",
              trainable="vae", unet_level="none", log_every=1000, save_every=0, resume=ck, preempt_file=pre)
    assert ck2.name == "geometry_step12.pt" and ck2.exists()
