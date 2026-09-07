"""scripts/smoke_test_atm_sc.py 의 synthetic 단계를 pytest 로도 돌린다.

integration 단계는 pretrained 체크포인트와 GPU 가 필요하므로 여기서는 제외한다
(스크립트를 직접 실행할 것).
"""
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_smoke_passes():
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "legacy" / "smoke_test_atm_sc.py"), "--skip-integration"],
        capture_output=True, text=True, timeout=600)
    assert "FINAL: PASS" in r.stdout, r.stdout[-3000:] + r.stderr[-2000:]
    assert "FAIL" not in r.stdout.replace("PASS/FAIL", ""), r.stdout[-3000:]


def test_synthetic_smoke_pass_mode():
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "legacy" / "smoke_test_atm_sc.py"),
         "--skip-integration", "--mode", "pass"],
        capture_output=True, text=True, timeout=600)
    assert "FINAL: PASS" in r.stdout, r.stdout[-3000:] + r.stderr[-2000:]


def test_spaces_selftest_runs_on_import():
    """spaces.py 는 import 시 축 순서/왕복 변환을 스스로 검사한다."""
    sys.path.insert(0, str(ROOT / "src"))
    import atm_sc.spaces as sp
    assert sp.W_SHAPE == (193, 229, 193)
    assert sp.W_AFFINE[0, 3] == -96.0


def test_no_recurrent_tracking_in_decoder():
    """decoder 가 point-by-point 재귀 구조가 아닌지 (배치 축이 독립인지) 확인.

    N개를 한 번에 넣은 결과와 하나씩 넣은 결과가 같아야 한다.
    """
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "stable" / "stable"))
    try:
        from model.model import ConvVAE
    except ImportError:
        import pytest
        pytest.skip("upstream model.py 없음")
    torch.manual_seed(0)
    vae = ConvVAE(3, 64, 512).eval()
    z, a = torch.randn(4, 64), torch.randn(4, 512)
    with torch.no_grad():
        batch = vae.decode(z, a)
        one = torch.cat([vae.decode(z[i:i + 1], a[i:i + 1]) for i in range(4)])
    assert batch.shape == (4, 3, 128)
    assert torch.allclose(batch, one, atol=1e-5), float((batch - one).abs().max())
