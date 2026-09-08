"""T1 -> white matter 확률 맵 (FreeSurfer white surface 대체).

FreeSurfer recon-all 은 subject 당 6~10시간(206명이면 10~14일)이고 이 머신에 설치도 되어 있지
않다. 우리 인코더는 3D UNet 이라 표면 메시를 받아도 결국 볼륨으로 되돌려야 하므로, ANTs Atropos
3-class 분할로 얻은 WM 확률 볼륨으로 대체한다 (subject 당 ~5분).

입력은 rigid 로 MNI152 에 정합된 W 격자 T1 이다 (전처리 프로토콜). rigid 는 개인 뇌 형태를
보존하므로 템플릿 뇌 마스크가 개인 뇌 경계와 정확히 맞지 않는다. 그래서 마스크를 안쪽으로
침식해서 쓴다 -- 피질 가장자리를 조금 잃지만 두개골/두피가 새어 들어와 Atropos 의 3-class
가정을 깨뜨리는 것보다 낫고, WM 은 뇌 안쪽이라 침식의 영향을 받지 않는다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..spaces import W_AFFINE, W_SHAPE
from .prepare_t1 import TEMPLATE_MASK, resample_to_W

N_CLASS = 3          # CSF / GM / WM (T1 강도 오름차순)


def _to_ants(vol: np.ndarray):
    import ants
    img = ants.from_numpy(np.ascontiguousarray(vol.astype(np.float32)))
    img.set_spacing((1.0, 1.0, 1.0))
    img.set_origin(tuple(float(v) for v in W_AFFINE[:3, 3]))
    return img


def brain_mask_W(erode_mm: int = 4) -> np.ndarray:
    """템플릿 뇌 마스크를 W 격자로 옮기고 erode_mm 만큼 침식.

    rigid 정합은 개인 뇌 형태를 보존하므로 템플릿 마스크가 뇌 경계와 어긋난다. 밖으로 넘치면
    두개골 지방(T1 에서 밝다)이 WM 으로 오분류되므로 안쪽으로 깎는다.
    """
    import nibabel as nib
    from scipy.ndimage import binary_erosion, binary_fill_holes
    m = resample_to_W(nib.load(str(TEMPLATE_MASK)), order=0) > 0.5
    # 템플릿 마스크는 뇌실을 구멍으로 남긴다(10,714 복셀). 그대로 침식하면 구멍이 66,329 로
    # 부풀어 뇌실 주변 백질까지 잘려나간다. 먼저 메우고 침식한다 -- 뇌실 자체는 Atropos 가
    # CSF 로 분류하므로 마스크에 포함해도 WM 오분류가 생기지 않는다.
    m = binary_fill_holes(m)
    if erode_mm > 0:
        r = int(erode_mm)
        zz, yy, xx = np.ogrid[-r:r + 1, -r:r + 1, -r:r + 1]
        ball = (zz ** 2 + yy ** 2 + xx ** 2) <= r ** 2      # 정육면체는 상자 모양 인공물을 만든다
        m = binary_erosion(m, ball)
    assert m.shape == W_SHAPE and m.sum() > 800_000, f"뇌 마스크가 이상하다: {m.sum()}"
    assert not (binary_fill_holes(m) & ~m).any(), "마스크에 내부 구멍이 남아 있다"
    return m


def wm_probability(t1_W: np.ndarray, mask: np.ndarray | None = None,
                   mrf: float = 0.2, iters: int = 5, return_all: bool = False):
    """W 격자 T1 -> WM 확률 [193,229,193] float32 (0~1).

    return_all=True 면 {"wm","gm","csf","brain"} dict 를 준다 (brain 은 subject 뇌 마스크 bool).

    Atropos 를 k-means(3) 로 초기화하고, 클래스 평균 강도가 가장 높은 것을 WM 으로 본다
    (T1 에서 WM > GM > CSF). 클래스 번호를 고정으로 가정하지 않는다.
    """
    import ants
    if mask is None:
        mask = brain_mask_W()
    assert t1_W.shape == W_SHAPE, t1_W.shape
    assert np.isfinite(t1_W).all() and t1_W.max() > 0, "T1 이 비었거나 NaN"

    # 분할 전에 [0,1] 로 정규화한다 (뇌 안 99.5 백분위 기준, 위는 clip).
    # 원 강도로 k-means 를 돌리면 밝은 이상치(혈관/지방/아티팩트)가 별도 클래스로 떨어져
    # "가장 밝은 클래스" 가 WM 이 아니라 0.3~2 % 짜리 잔여물이 된다 (206명 중 5명 실패).
    # 백분위 clip 으로 밝은 꼬리를 눌러야 3-class 가 CSF/GM/WM 로 갈린다.
    scale = float(np.percentile(t1_W[mask], 99.5))
    assert scale > 0, "뇌 안 T1 백분위가 0 -- 정합 실패 의심"
    t1_n = np.clip(t1_W / scale, 0.0, 1.0).astype(np.float32)

    img, msk = _to_ants(t1_n), _to_ants(mask.astype(np.float32))
    seg = ants.atropos(a=img, x=msk, i=f"kmeans[{N_CLASS}]",
                       m=f"[{mrf},1x1x1]", c=f"[{iters},0]")
    probs = [p.numpy() for p in seg["probabilityimages"]]
    assert len(probs) == N_CLASS, f"확률 맵 {len(probs)}개 (기대 {N_CLASS})"
    # 각 클래스의 가중 평균 강도로 WM 을 고르되, **크기가 그럴듯한 클래스만 후보**로 본다.
    # 정규화 후에도 밝은 꼬리가 남는 subject 가 있어(206명 중 2명) 최고 강도 클래스가
    # 3~5 % 짜리 잔여물이 되는 경우가 있다. WM 은 뇌의 30~45 % 이므로 하한을 둔다.
    means = np.array([float((t1_n * p).sum() / max(p.sum(), 1e-9)) for p in probs])
    sizes = np.array([float((p > 0.5).sum()) / float(mask.sum()) for p in probs])
    cand = np.where(sizes >= 0.15)[0]
    k = int(cand[np.argmax(means[cand])]) if len(cand) else int(np.argmax(means))
    wm = probs[k].astype(np.float32)
    assert wm.shape == W_SHAPE, wm.shape
    frac = float((wm > 0.5).sum()) / float(mask.sum())
    assert 0.20 < frac < 0.70, f"WM 비율이 이상하다: {frac:.3f} (뇌 안에서 20~70% 여야 함)"
    assert np.isfinite(wm).all() and wm.max() > 0.9, "WM 확률이 비었거나 NaN"
    if not return_all:
        return wm
    # CSF < GM < WM (T1 강도 오름차순). WM 은 위에서 크기 하한까지 걸어 고른 k 를 쓰고,
    # 나머지 둘은 평균 강도 순으로 가른다. 예전에는 WM 만 저장하고 GM/CSF 를 버렸는데,
    # ROI 별 GM 부피와 CSF(위축) 는 자로 잰 개인차 feature 라 버릴 이유가 없었다.
    rest = [c for c in range(N_CLASS) if c != k]
    csf_i, gm_i = sorted(rest, key=lambda c: means[c])
    out = {"wm": wm, "gm": probs[gm_i].astype(np.float32), "csf": probs[csf_i].astype(np.float32)}
    tissue = out["wm"] + out["gm"] + out["csf"]
    assert float(np.abs(tissue[mask] - 1.0).max()) < 0.05, "3 class 확률 합이 1 이 아니다"
    # subject 뇌 마스크: 템플릿 침식본이 아니라 이 subject 의 조직이 실제로 있는 곳.
    # 침식 마스크 밖은 Atropos 가 보지 않았으므로 0 이고, 그래서 침식 마스크와의 AND 가 된다.
    out["brain"] = ((out["gm"] + out["wm"]) > 0.5) & mask
    assert out["brain"].sum() > 500_000, f"subject 뇌 마스크가 너무 작다: {int(out['brain'].sum())}"
    return out


def wm_path(cache: Path, sub: str) -> Path:
    return Path(cache) / f"{sub}_WM_W.npy"


def tissue_path(cache: Path, sub: str, source: str = "rigid") -> Path:
    """CSF/GM/WM 확률 + subject 뇌 마스크 (uint8 로 압축 저장, 1/255 단위).

    두 벌이 필요하다.
      rigid  개인 뇌 형태를 보존한다 -> **feature** 용 (형태 자체가 개인차 신호다)
      syn    NLin6 로 비선형 정합된다 -> **filtering/trimming** 용. GT 가닥이 QSDR/NLin6 공간에
             있어서 rigid 조직맵과는 조직 경계가 어긋난다 (측정: GT 점이 GM+WM 안에 들어가는
             비율 rigid 0.820 / syn 0.907, GT WM 점유 중앙값 0.573 / 0.801).
    """
    tag = "" if source == "rigid" else f"_{source}"
    return Path(cache) / f"{sub}_tissue{tag}_W.npz"
