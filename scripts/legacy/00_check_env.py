#!/usr/bin/env python
"""환경/업스트림 회귀 점검. 다른 스크립트를 돌리기 전에 이것부터 통과시킨다."""
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from atm_sc.models.atm_adapter import ATMBundle, BundleNorm, UPSTREAM, BUNDLES  # noqa: E402
from atm_sc.spaces import W_SHAPE                                               # noqa: E402

ok = True


def check(name, fn):
    global ok
    try:
        print(f"  {name:42s} {fn()}")
    except Exception as e:
        ok = False
        print(f"  {name:42s} FAIL  {type(e).__name__}: {str(e)[:90]}")


print("패키지")
check("torch / cuda", lambda: f"{torch.__version__}  cuda={torch.cuda.is_available()}")
check("cudnn (conv3d 동작 여부)", lambda: (
    torch.nn.Conv3d(1, 8, 3, padding=1).cuda()(torch.randn(1, 1, 16, 16, 16, device="cuda")).shape,
    torch.backends.cudnn.version())[1])
for m in ("numpy", "scipy", "nibabel", "sklearn", "joblib", "dipy", "ants"):
    check(m, lambda m=m: __import__(m).__version__)

print("\nupstream 자산")
check("model.py / infer.py", lambda: all((UPSTREAM / f).exists() for f in ("model/model.py", "infer.py")))
check("supp/*.npy (D4: data/ 아님)", lambda: len(list((UPSTREAM / "supp").glob("*.npy"))))
have = [b for b in BUNDLES if (UPSTREAM / "models" / b / f"atmvae_{b}.pth").exists()]
check("체크포인트", lambda: f"{len(have)}/30  {have if len(have) < 5 else ''}")
check("KDE", lambda: f"{sum((UPSTREAM/'kde_models'/b/'kde_model.joblib').exists() for b in BUNDLES)}/30")

print("\n데이터")
R = Path(__file__).resolve().parents[1]
check("atlas 82 ROI", lambda: int(np.asanyarray(__import__("nibabel").load(
    R / "DesikanCortexPD25_space-MNI152NLin6_res-2x2x2.nii.gz").dataobj).max()))
check("GT SC (.mat) subject 수", lambda: len(__import__("scipy.io", fromlist=["io"]).loadmat(
    R / "FC_DKPD25_82_ppmi_all_nomed_qc.mat")["data"][0]))
check("MNI152NLin6 템플릿", lambda: (R / "templates" / "tpl-MNI152NLin6Asym_res-01_T1w.nii.gz").exists())

if have:
    b = have[0]
    print(f"\n인코더 전용 경로 == upstream 전체 forward ({b}, CPU)")
    atm = ATMBundle(b, BundleNorm.from_upstream(b), device="cpu")
    x = torch.randn(1, 1, *W_SHAPE) * 0.3 + 0.5
    t = time.time(); a_full = atm.encode_anatomy(x, full=True); t_full = time.time() - t
    t = time.time(); a_enc = atm.encode_anatomy(x, full=False); t_enc = time.time() - t
    d = float((a_full - a_enc).abs().max())
    print(f"  전체 {t_full:.0f}s / 인코더만 {t_enc:.0f}s   최대 절대 오차 = {d:.3e}")
    assert d == 0.0, f"인코더 전용 경로가 upstream 과 다름 ({d})"
    print("  동일 (bit-exact)")

print("\nOK" if ok else "\n일부 항목 실패")
sys.exit(0 if ok else 1)
