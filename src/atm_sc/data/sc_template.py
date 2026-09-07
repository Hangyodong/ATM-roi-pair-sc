"""train split 전용 SC 통계 — log1p 템플릿, edge 별 표준편차, edge mask.

왜 별도 모듈인가: 이 세 값은 **train subject 로만** 계산해야 하고 (val/test 가 들어가면
잔차 타깃 자체가 누수다), checkpoint 와 함께 저장해서 평가 때 같은 값을 써야 한다.
모델이 이미 갖고 있는 `model.template["sc_pass"]` 는 **count 공간** 평균이라 여기서 쓸 수 없다
(log1p 평균과 다르다 -- Jensen).

전략 문서 §1.2~1.4.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .dataset import ROIPairSubject

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "outputs" / "cache"


def build(subjects: list[str], std_floor: float = 0.1, min_prevalence: float = 0.0) -> dict:
    """[N_train, R, R] GT SC -> upper-triangle 통계.

    반환: template [E] (log1p 평균), std [E] (log1p 표준편차, floor 적용 전 raw 도 함께),
          mask [E] bool (잔차 손실에 쓸 edge), prevalence [E].
    """
    assert len(subjects) >= 10, f"템플릿에 train subject 10명 이상 필요 (now {len(subjects)})"
    Y, n_roi = [], None
    for s in subjects:
        w = np.asarray(ROIPairSubject(s).sc_mat, np.float64)
        assert w.ndim == 2 and w.shape[0] == w.shape[1], (s, w.shape)
        assert np.isfinite(w).all(), f"{s}: GT SC 에 NaN/Inf"
        assert w.sum() > 0, f"{s}: GT SC 가 전부 0"
        n_roi = w.shape[0] if n_roi is None else n_roi
        assert w.shape[0] == n_roi, (s, w.shape, n_roi)
        iu = np.triu_indices(n_roi, 1)
        Y.append(np.log1p(w[iu]))
    Y = np.stack(Y)                                   # [N, E]
    template = Y.mean(0)
    std_raw = Y.std(0, ddof=1)
    prevalence = (Y > 0).mean(0)
    mask = (std_raw > std_floor * 0.5) & (prevalence >= min_prevalence)
    assert mask.sum() > 100, f"edge mask 가 너무 좁다 ({int(mask.sum())}개) -- std_floor 를 낮춰라"
    out = {"template": template.astype(np.float32),
           "std": np.maximum(std_raw, std_floor).astype(np.float32),
           "std_raw": std_raw.astype(np.float32),
           "prevalence": prevalence.astype(np.float32),
           "mask": mask,
           "n_roi": np.int64(n_roi),
           "n_subjects": np.int64(len(subjects)),
           "std_floor": np.float32(std_floor),
           "subjects": np.array(subjects, dtype="U16")}
    # 잔차가 실재하는지 -- 전부 0 이면 배울 개인차가 없다는 뜻이라 여기서 멈춘다.
    resid = (Y - template) / out["std"]
    out["resid_share"] = np.float32(((Y - template).var()) / Y.var())
    assert float(resid.var()) > 1e-3, "정규화 잔차 분산이 0 -- subject 가 전부 같다"
    return out


def load_or_build(subjects: list[str], path: Path | None = None, **kw) -> dict:
    """path 가 있으면 읽고, 없으면 만들어 저장한다. subject 목록이 다르면 다시 만든다."""
    path = Path(path) if path is not None else CACHE / "sc_template_stats.npz"
    if path.exists():
        z = np.load(path, allow_pickle=False)
        if list(z["subjects"]) == list(np.array(subjects, dtype="U16")):
            return {k: z[k] for k in z.files}
        print(f"[sc_template] subject 목록이 달라 재계산 ({len(z['subjects'])} -> {len(subjects)})", flush=True)
    st = build(subjects, **kw)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **st)
    print(f"[sc_template] 저장: {path} (train {len(subjects)}명, edge {int(st['mask'].sum())}/"
          f"{st['template'].size}, 잔차 비중 {float(st['resid_share']):.3f})", flush=True)
    return st
