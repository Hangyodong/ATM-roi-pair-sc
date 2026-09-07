"""데이터 경로 해석. 전체 압축 해제(data/) 와 probe(ppmi_probe/) 둘 다 지원한다."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ATLAS = ROOT / "DesikanCortexPD25_space-MNI152NLin6_res-2x2x2.nii.gz"
SC_MAT = ROOT / "FC_DKPD25_82_ppmi_all_nomed_qc.mat"
CACHE = ROOT / "outputs" / "cache"
_ROOTS = [ROOT / "PPMI_QC263_tracto" / "PPMI_QC263_tracto",
          ROOT / "data" / "PPMI_QC263_tracto",
          ROOT / "ppmi_probe" / "PPMI_QC263_tracto"]


def subject_dir(sub: str) -> Path:
    for r in _ROOTS:
        if (r / sub).is_dir():
            return r / sub
    raise FileNotFoundError(f"{sub} 를 찾을 수 없음. 확인한 위치: {[str(r) for r in _ROOTS]}")


def t1_path(sub: str) -> Path:
    return subject_dir(sub) / f"{sub}_T1w.nii.gz"


def tt_path(sub: str) -> Path:
    return subject_dir(sub) / f"{sub}_tract.tt.gz"


def manifest() -> Path:
    for r in _ROOTS:
        if (r / "manifest.csv").exists():
            return r / "manifest.csv"
    raise FileNotFoundError("manifest.csv 없음")


def mat_subjects() -> list[str]:
    """GT SC 가 있는 subject = FC_DKPD25_*.mat 의 data.subject (238명, dwi_qc==pass 만 수록)."""
    import scipy.io as sio
    m = sio.loadmat(SC_MAT, variable_names=["data"])
    subs = [str(r["subject"][0]) for r in m["data"][0]]
    assert len(subs) == len(set(subs)), ".mat 에 중복 subject"
    return subs


def folder_subjects() -> list[str]:
    """tracto 폴더에 T1 과 .tt.gz 가 **둘 다** 있는 subject."""
    out = []
    for r in _ROOTS:
        if not r.is_dir():
            continue
        for d in sorted(p.name for p in r.iterdir() if p.name.startswith("sub-") and p.is_dir()):
            if (r / d / f"{d}_T1w.nii.gz").exists() and (r / d / f"{d}_tract.tt.gz").exists():
                out.append(d)
    return sorted(set(out))


def subjects(require_sc: bool = True) -> list[str]:
    """학습/평가에 쓸 subject = **.mat(GT SC) ∩ tracto 폴더(T1 + .tt.gz)**.

    .mat 에 없는 subject 는 GT SC 가 없으므로(dwi_qc FAIL) 학습에 쓸 수 없고,
    폴더에 없는 subject 는 T1/tractogram 이 없다. 둘 다 있는 subject 만 돌려준다.
    require_sc=False 면 폴더 기준 전체 (QC 스크립트 등 SC 가 필요 없는 용도).
    """
    folder = folder_subjects()
    if not require_sc:
        return folder
    return sorted(set(mat_subjects()) & set(folder))


def subject_overlap_report() -> dict:
    mat, folder = set(mat_subjects()), set(folder_subjects())
    return {"mat": len(mat), "folder": len(folder), "both": len(mat & folder),
            "mat_only": sorted(mat - folder), "folder_only": sorted(folder - mat)}


def subject_meta() -> dict:
    """subject -> {group, batch, batch2, proto, age, sex}. scanner/site 는 .mat 에만 있다 (manifest 에는 없음)."""
    import scipy.io as sio
    m = sio.loadmat(SC_MAT, variable_names=["data"])
    out = {}
    for r in m["data"][0]:
        g = lambda k: str(r[k].ravel()[0]) if r[k].size else ""
        out[g("subject")] = {"group": g("group"), "batch": g("batch"), "batch2": g("batch2"),
                             "proto": g("proto"), "age": g("age"), "sex": g("sex")}
    return out
