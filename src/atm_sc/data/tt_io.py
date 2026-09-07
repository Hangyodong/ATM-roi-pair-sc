"""DSI Studio TinyTrack (.tt.gz) 읽기 + GT 라벨/행렬 생성.

PPMI 데이터의 tractogram 은 .trk 가 아니라 .tt.gz 다. gzip 으로 감싼 MATLAB v4
파일이고 streamline 은 1/32 voxel 정밀도의 int8 delta 로 인코딩되어 있다.

레코드 하나:
    [uint32 size = 3*npts][int32 x,y,z (x32)][int8 dx,dy,dz] * (npts-1)
따라서 레코드 총 바이트 = size + 13.

GT SC 는 endpoint 가 아니라 **pass** (streamline 이 통과한 ROI 집합의 모든 쌍) 로
계산되어 있다. 실측: pass r=0.9986 / end r=0.673 (vs .mat 의 SC_weight).
"""
from __future__ import annotations

import gzip
import io
from dataclasses import dataclass

import numpy as np
import scipy.io as sio

from ..spaces import apply_affine

TT_SCALE = 32.0


@dataclass
class TTHeader:
    dimension: np.ndarray      # (3,) int
    voxel_size: np.ndarray     # (3,) float
    trans_to_mni: np.ndarray   # (4,4)  tt voxel -> MNI mm
    report: str

    def to_mm(self, vox: np.ndarray) -> np.ndarray:
        return apply_affine(self.trans_to_mni, vox)


def read_header(path: str) -> TTHeader:
    with gzip.open(path, "rb") as f:
        raw = f.read()
    d = sio.loadmat(io.BytesIO(raw),
                    variable_names=["dimension", "voxel_size", "trans_to_mni", "report"])
    return _mk_header(d)


def _mk_header(d) -> TTHeader:
    T = np.asarray(d["trans_to_mni"], np.float64).reshape(4, 4)
    assert abs(np.linalg.det(T[:3, :3])) > 1e-6, "trans_to_mni 가 특이행렬"
    rep = bytes(np.asarray(d["report"]).ravel().astype(np.uint8)).decode("ascii", "replace") \
        if "report" in d else ""
    return TTHeader(np.asarray(d["dimension"]).ravel().astype(int),
                    np.asarray(d["voxel_size"]).ravel().astype(float), T, rep)


def _scan_records(buf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """레코드 시작 오프셋(size 필드 뒤, int32 헤더 위치)과 점 개수를 훑는다.

    각 레코드의 길이가 앞 레코드에 의존하므로 이 스캔만은 순차적일 수밖에 없다.
    끝에서 버퍼를 정확히 소진하지 못하면 포맷 가정이 틀린 것이므로 즉시 중단한다.
    """
    raw = buf.tobytes()
    n = len(raw)
    offs, npts = [], []
    i = 0
    while i < n:
        size = int.from_bytes(raw[i:i + 4], "little")
        assert size >= 3 and size % 3 == 0, f"레코드 size 이상: {size} @ {i}"
        i += 4
        offs.append(i)
        npts.append(size // 3)
        i += 12 + (size - 3)
    assert i == n, f"버퍼를 정확히 소진하지 못함: {i} != {n} (.tt 포맷 가정 오류)"
    return np.asarray(offs, np.int64), np.asarray(npts, np.int64)


def _ragged_indices(starts: np.ndarray, lens: np.ndarray) -> np.ndarray:
    """concat([arange(s, s+l) for s, l in zip(starts, lens)]) 의 벡터화 구현."""
    total = int(lens.sum())
    if total == 0:
        return np.empty(0, np.int64)
    out = np.ones(total, np.int64)
    ends = np.cumsum(lens)
    out[0] = starts[0]
    if len(starts) > 1:
        out[ends[:-1]] = starts[1:] - (starts[:-1] + lens[:-1]) + 1
    return np.cumsum(out, out=out)


def _decode_chunk(buf: np.ndarray, offs: np.ndarray, npts: np.ndarray) -> np.ndarray:
    """레코드 묶음을 (sum(npts), 3) int64 (1/32 voxel 단위) 로 디코딩."""
    n = len(offs)
    hdr = buf[_ragged_indices(offs, np.full(n, 12, np.int64))]
    first = np.ascontiguousarray(hdr).view("<i4").reshape(n, 3).astype(np.int64)

    P = int(npts.sum())
    D = np.zeros((P, 3), np.int64)
    seg_start = np.concatenate([[0], np.cumsum(npts)[:-1]])
    D[seg_start] = first

    dlen = 3 * (npts - 1)
    if dlen.sum():
        didx = _ragged_indices(offs + 12, dlen)
        deltas = buf[didx].view("<i1").astype(np.int64).reshape(-1, 3)
        mask = np.ones(P, bool)
        mask[seg_start] = False
        D[mask] = deltas

    C = np.cumsum(D, axis=0)
    base = np.repeat(first - C[seg_start], npts, axis=0)
    return C + base


def _decode_chunk_reference(buf: np.ndarray, offs, npts) -> np.ndarray:
    """느리지만 명백하게 맞는 구현. 벡터화 버전 검증용."""
    out = []
    for o, npt in zip(offs, npts):
        a = np.frombuffer(buf[o:o + 12].tobytes(), "<i4").astype(np.int64)
        if npt > 1:
            d = np.frombuffer(buf[o + 12:o + 12 + 3 * (npt - 1)].tobytes(), "<i1")
            out.append(np.vstack([a, a + np.cumsum(d.astype(np.int64).reshape(-1, 3), 0)]))
        else:
            out.append(a[None])
    return np.concatenate(out)


def load_streamlines(path: str, chunk: int = 200_000, verify: bool = True):
    """.tt.gz -> (header, generator of (points_mm [P,3] float32, npts [n] int64)).

    subject 하나가 1,000,000 streamline / 6600 만 점이라 전부 메모리에 올리지 않고
    chunk 단위로 넘긴다.
    """
    with gzip.open(path, "rb") as f:
        raw = f.read()
    hdr = _mk_header(sio.loadmat(io.BytesIO(raw),
                     variable_names=["dimension", "voxel_size", "trans_to_mni", "report"]))
    d = sio.loadmat(io.BytesIO(raw), variable_names=["track", "track1"])
    keys = [k for k in ("track", "track1") if k in d]
    assert keys, f"{path}: track 변수 없음"

    def gen():
        checked = not verify
        for k in keys:
            buf = np.asarray(d[k]).ravel().view(np.uint8)
            offs, npts = _scan_records(buf)
            for c in range(0, len(offs), chunk):
                o, p = offs[c:c + chunk], npts[c:c + chunk]
                V = _decode_chunk(buf, o, p)
                if not checked:                      # 첫 200개로 벡터화 구현 대조
                    m = min(200, len(o))
                    ref = _decode_chunk_reference(buf, o[:m], p[:m])
                    assert np.array_equal(V[:int(p[:m].sum())], ref), "벡터화 디코더 불일치"
                    checked = True
                mm = hdr.to_mm(V / TT_SCALE).astype(np.float32)
                assert np.isfinite(mm).all(), "디코딩 결과에 NaN/Inf"
                yield mm, p
    return hdr, gen


# --- streamline 유도량 -------------------------------------------------------

def segment_lengths(mm: np.ndarray, npts: np.ndarray) -> np.ndarray:
    """streamline 별 길이 (mm). resample 전에 계산해야 한다."""
    tid = np.repeat(np.arange(len(npts)), npts)
    seg = np.linalg.norm(np.diff(mm, axis=0), axis=1)
    same = tid[1:] == tid[:-1]
    return np.bincount(tid[1:][same], weights=seg[same], minlength=len(npts))


def resample_128(mm: np.ndarray, npts: np.ndarray, n_out: int = 128) -> np.ndarray:
    """가변 길이 streamline -> [n, n_out, 3] 등간격 재샘플 (ATM 입력 형식)."""
    starts = np.concatenate([[0], np.cumsum(npts)[:-1]])
    out = np.empty((len(npts), n_out, 3), np.float32)
    tgt = np.linspace(0.0, 1.0, n_out)
    for i, (s, p) in enumerate(zip(starts, npts)):
        pts = mm[s:s + p]
        if p == 1:
            out[i] = pts
            continue
        d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
        if d[-1] <= 0:
            out[i] = pts[0]
            continue
        u = d / d[-1]
        for a in range(3):
            out[i, :, a] = np.interp(tgt, u, pts[:, a])
    return out


def point_labels(mm: np.ndarray, atlas: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """점 -> ROI 라벨 (0 = 배경). nearest neighbour."""
    inv = np.linalg.inv(affine)
    ijk = np.rint(apply_affine(inv, mm.astype(np.float64))).astype(np.int64)
    shp = np.asarray(atlas.shape)
    ok = np.all((ijk >= 0) & (ijk < shp), axis=1)
    lab = np.zeros(len(ijk), np.int16)
    lab[ok] = atlas[ijk[ok, 0], ijk[ok, 1], ijk[ok, 2]]
    return lab


def roi_visit_sets(lab: np.ndarray, npts: np.ndarray):
    """streamline 별 통과 ROI 집합. (track_idx, roi_idx0based) 정렬 배열을 돌려준다."""
    tid = np.repeat(np.arange(len(npts)), npts)
    keep = lab > 0
    pk = np.stack([tid[keep], lab[keep].astype(np.int64) - 1], 1)
    return np.unique(pk, axis=0)


def hard_sc(mm, npts, atlas, affine, n_roi: int, mode: str = "pass"):
    """GT 정의대로 SC weight / mean length 계산. mode: 'pass' | 'end'."""
    assert mode in ("pass", "end"), mode
    W = np.zeros((n_roi, n_roi), np.int64)
    S = np.zeros((n_roi, n_roi), np.float64)
    lab = point_labels(mm, atlas, affine)
    L = segment_lengths(mm, npts)
    if mode == "end":
        starts = np.concatenate([[0], np.cumsum(npts)[:-1]])
        a, b = lab[starts], lab[starts + npts - 1]
        m = (a > 0) & (b > 0) & (a != b)
        u, v = a[m] - 1, b[m] - 1
        np.add.at(W, (u, v), 1); np.add.at(W, (v, u), 1)
        np.add.at(S, (u, v), L[m]); np.add.at(S, (v, u), L[m])
    else:
        pk = roi_visit_sets(lab, npts)
        bnd = np.searchsorted(pk[:, 0], np.arange(len(npts) + 1))
        for t in range(len(npts)):
            rs = pk[bnd[t]:bnd[t + 1], 1]
            if len(rs) < 2:
                continue
            ii, jj = np.triu_indices(len(rs), 1)
            u, v = rs[ii], rs[jj]
            np.add.at(W, (u, v), 1); np.add.at(W, (v, u), 1)
            np.add.at(S, (u, v), L[t]); np.add.at(S, (v, u), L[t])
    return W, S
