"""tractolearn (external/tractolearn, scil-vital, EULA: 학술/비상업 사용) 소스를 현재 환경에서 import 가능하게 한다.

tractolearn 은 torch 1.13 / numpy 1.23 / dipy 1.7 / scilpy(git) 에 고정돼 있어 pip 로 설치하지 않는다.
대신 이 모듈을 먼저 import 하면:
  1. external/tractolearn 을 sys.path 에 넣고
  2. dipy 1.9+ 에서 사라진 `dipy.tracking.metrics.downsample` 을 `set_number_of_points` 로 되살리고
  3. scilpy 가 없으면 `scilpy.segment.streamlines.streamlines_in_mask` 를 dipy 로 구현한 대체 모듈을 등록하고
  4. fury(VTK 렌더링, 설치하면 numpy 가 2.4 로 올라감) 가 없으면 import 만 통과하는 stub 을 등록한다
     (`learning.trainer_manager` 가 plot 용으로 import; 실제 렌더링 호출 시 NotImplementedError).
tractolearn 소스 자체는 수정하지 않는다.

    from atm_sc.compat import tractolearn_env          # noqa: F401  (부수효과)
    from tractolearn.models.model_pool import get_model
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TRACTOLEARN = ROOT / "external" / "tractolearn"
assert (TRACTOLEARN / "tractolearn" / "__init__.py").exists(), f"tractolearn 소스 없음: {TRACTOLEARN}"
if str(TRACTOLEARN) not in sys.path:
    sys.path.insert(0, str(TRACTOLEARN))

# --- dipy: downsample -> set_number_of_points (같은 의미: 등간격 n 점 재샘플) ---------------------
import dipy.tracking.metrics as _metrics  # noqa: E402

if not hasattr(_metrics, "downsample"):
    from dipy.tracking.streamline import set_number_of_points as _snp

    def downsample(xyz, n_pols=3):
        return _snp(xyz, nb_points=n_pols)

    _metrics.downsample = downsample

# --- scilpy: streamlines_in_mask 만 tractolearn 이 쓴다 -----------------------------------------
try:
    import scilpy.segment.streamlines  # noqa: F401
except ImportError:
    import numpy as np
    from dipy.io.stateful_tractogram import Space

    def streamlines_in_mask(sft, target_mask, all_in=False):
        """scilpy.segment.streamlines.streamlines_in_mask 와 같은 반환: 각 streamline 이 mask 에 걸치는지
        (all_in=True 면 모든 점이 mask 안) bool 배열. sft 는 StatefulTractogram, mask 는 sft 와 같은 격자."""
        sft.to_vox(); sft.to_corner()
        mask = np.asarray(target_mask).astype(bool)
        out = np.zeros(len(sft.streamlines), dtype=bool)
        for i, s in enumerate(sft.streamlines):
            v = np.floor(s).astype(int)
            inside = (v >= 0).all(1) & (v < mask.shape).all(1)
            hit = np.zeros(len(s), dtype=bool)
            hit[inside] = mask[v[inside, 0], v[inside, 1], v[inside, 2]]
            out[i] = hit.all() if all_in else hit.any()
        return out

    _pkg = types.ModuleType("scilpy"); _pkg.__path__ = []
    _seg = types.ModuleType("scilpy.segment"); _seg.__path__ = []
    _mod = types.ModuleType("scilpy.segment.streamlines")
    _mod.streamlines_in_mask = streamlines_in_mask
    _pkg.segment = _seg; _seg.streamlines = _mod
    sys.modules.update({"scilpy": _pkg, "scilpy.segment": _seg, "scilpy.segment.streamlines": _mod})
    _ = Space

# --- fury: 시각화 전용. import 만 통과시킨다 ---------------------------------------------------
try:
    import fury  # noqa: F401
except ImportError:
    class _NoFury:
        """속성 접근은 무한히 허용(예: `window.colors.black` 이 기본 인자로 쓰임), 호출하면 실패."""
        def __init__(self, name): self._name = name
        def __getattr__(self, item):
            return _NoFury(f"{self._name}.{item}")
        def __call__(self, *a, **k):
            raise NotImplementedError(f"{self._name}: fury(VTK) 미설치 — 렌더링은 이 환경에서 지원하지 않음")

    _fury = types.ModuleType("fury"); _fury.__path__ = []
    _fury.actor, _fury.window = _NoFury("fury.actor"), _NoFury("fury.window")
    _cmap = types.ModuleType("fury.colormap"); _cmap.line_colors = _NoFury("fury.colormap.line_colors")
    _fury.colormap = _cmap
    sys.modules.update({"fury": _fury, "fury.colormap": _cmap})
