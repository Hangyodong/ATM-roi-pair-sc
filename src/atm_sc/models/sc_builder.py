"""미분 가능한 SC / tract-length builder.

training loss graph 에 tck2connectome 같은 외부 CLI 나 hard atlas lookup 을 넣지 않는다.

mode='endpoint'  (Framework v2 §8.1)
    SC = sum_k 0.5 * (q_s ⊗ q_e + q_e ⊗ q_s) = 0.5 * (Qs^T Qe + Qe^T Qs)
mode='pass'
    SC = sum_k u_k ⊗ u_k = U^T U   (대각 제거)
    이 프로젝트의 GT SC 가 pass 정의이므로 GT 재현에는 이쪽을 쓴다.

둘 다 matmul 두 번이라 streamline 수에 선형이고, 무엇보다 **streamline 에 대해 가산적**
이다. 따라서 chunk 로 나눠 부분 SC 를 더한 뒤 마지막에 한 번만 loss 를 계산할 수 있고
(ChunkedSC), 30 bundle 에 대해서도 같은 성질로 정확한 gradient 를 누적할 수 있다
(BundleAccumulator).
"""
from __future__ import annotations

import torch


def streamline_lengths(mm: torch.Tensor) -> torch.Tensor:
    """[N,T,3] -> [N] 길이(mm)."""
    return torch.linalg.norm(mm[:, 1:] - mm[:, :-1], dim=-1).sum(dim=1)


def _drop_diag(x: torch.Tensor) -> torch.Tensor:
    return x - torch.diag_embed(torch.diagonal(x))


def endpoint_sc(q_start: torch.Tensor, q_end: torch.Tensor,
                lengths: torch.Tensor | None = None, weights: torch.Tensor | None = None):
    """pipeline §14/§16. (SC [R,R], Num [R,R] 또는 None). 정의상 대칭.

    SC(i,j)  = sum_k w_k * 0.5 * (q_s(k,i) q_e(k,j) + q_e(k,i) q_s(k,j))
    Num(i,j) = 같은 식에 L_k 를 곱한 것  -> Length = Num / SC
    weights 가 None 이면 w_k = 1 (순수 count).
    """
    assert q_start.shape == q_end.shape, (q_start.shape, q_end.shape)
    n = q_start.shape[0]
    w = torch.ones(n, device=q_start.device, dtype=q_start.dtype) if weights is None else weights
    assert w.shape == (n,), w.shape
    qs_w = q_start * w[:, None]
    sc = _drop_diag(0.5 * (qs_w.T @ q_end + q_end.T @ qs_w))
    if lengths is None:
        return sc, None
    assert lengths.shape == (n,), lengths.shape
    qs_wl = q_start * (w * lengths)[:, None]
    num = _drop_diag(0.5 * (qs_wl.T @ q_end + q_end.T @ qs_wl))
    return sc, num


def pass_sc(u: torch.Tensor, lengths: torch.Tensor | None = None,
            weights: torch.Tensor | None = None):
    """(SC [R,R], Num [R,R] 또는 None). u^T diag(w) u 는 대칭."""
    assert u.ndim == 2, u.shape
    n = u.shape[0]
    w = torch.ones(n, device=u.device, dtype=u.dtype) if weights is None else weights
    assert w.shape == (n,), w.shape
    sc = _drop_diag((u * w[:, None]).T @ u)
    if lengths is None:
        return sc, None
    assert lengths.shape == (n,), lengths.shape
    return sc, _drop_diag((u * (w * lengths)[:, None]).T @ u)


class SCBuilder(torch.nn.Module):
    """assigner + 집계 규칙. streamline 좌표 [N,T,3] -> (SC, Num)."""

    def __init__(self, assigner, mode: str = "endpoint"):
        super().__init__()
        assert mode in ("endpoint", "pass"), mode
        self.assigner, self.mode = assigner, mode
        self.n_roi = assigner.n_roi

    def forward(self, mm: torch.Tensor, lengths: torch.Tensor | None = None,
                weights: torch.Tensor | None = None):
        if lengths is None:
            lengths = streamline_lengths(mm)
        if self.mode == "endpoint":
            qs, qe = self.assigner.endpoint_probs(mm)
            return endpoint_sc(qs, qe, lengths, weights)
        return pass_sc(self.assigner.visit_probs(mm), lengths, weights)


class ChunkedSC:
    """VRAM 이 부족할 때 streamline 을 chunk 로 나눠 부분 SC 를 더한다.

    SC 는 streamline 에 대해 가산적이므로 chunk 별로 loss 를 걸면 안 되고
    (상관/로그가 비선형) 전부 더한 뒤 한 번만 계산해야 한다 (v2 §18).

        acc = ChunkedSC(n_roi, device)
        for chunk in chunks: acc += builder(chunk)
        loss = loss_fn(acc.sc, acc.num)
    """

    def __init__(self, n_roi: int, device="cuda", dtype=torch.float32):
        z = torch.zeros((n_roi, n_roi), dtype=dtype, device=device)
        self.sc, self.num = z, z.clone()

    def __iadd__(self, sc_num):
        sc, num = sc_num
        self.sc = self.sc + sc
        if num is not None:
            self.num = self.num + num
        return self


class BundleAccumulator:
    """30 bundle 을 한 그래프에 못 올릴 때 쓰는 2-pass 정확 gradient 누적.

    SC_total = sum_b SC_b 가 가산적이므로 dL/dSC_b = dL/dSC_total 이다. 근사 없음.
    forward 를 두 번 하는 대신 bundle 하나 분량의 그래프만 메모리에 올린다.
    """

    def __init__(self, n_roi: int, device="cuda", dtype=torch.float32):
        z = lambda: torch.zeros((n_roi, n_roi), dtype=dtype, device=device)
        self.sc, self.num = z(), z()

    def add(self, sc: torch.Tensor, num: torch.Tensor | None = None) -> None:
        self.sc += sc.detach()
        if num is not None:
            self.num += num.detach()

    def loss_grads(self, loss_fn):
        """loss_fn(SC, Num) -> scalar. (loss, dL/dSC, dL/dNum)."""
        sc = self.sc.detach().requires_grad_(True)
        num = self.num.detach().requires_grad_(True)
        loss = loss_fn(sc, num)
        assert loss.ndim == 0 and torch.isfinite(loss), loss
        loss.backward()
        gs = sc.grad if sc.grad is not None else torch.zeros_like(sc)
        gn = num.grad if num.grad is not None else torch.zeros_like(num)
        return float(loss.detach()), gs.detach(), gn.detach()

    @staticmethod
    def backward_bundle(sc, num, g_sc, g_num) -> None:
        t = (sc * g_sc).sum()
        if num is not None:
            t = t + (num * g_num).sum()
        t.backward()
