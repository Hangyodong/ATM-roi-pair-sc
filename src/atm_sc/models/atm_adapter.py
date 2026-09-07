"""ATM (external/atm_upstream) 래퍼.

upstream 은 한 줄도 고치지 않는다. 대신 infer.py 의 알려진 결함을 여기서 우회한다.

  D2  infer.py:132  .eval() 미호출 -> UNet Dropout3d 4개가 살아있어 anatomy feature 가
                    실행마다 달라진다. 여기서는 항상 eval() 을 강제하고, 원본 동작을
                    재현해야 할 때만 train_mode=True 로 명시적으로 켠다.
  D3  infer.py:140  anatomical_condition.repeat(3000, 1) 하드코딩 -> N != 3000 이면
                    FiLM 에서 shape 오류. 실제 N 으로 repeat 한다.
  D4  infer.py:146  좌표 상수를 data/ 에서 찾지만 파일은 supp/ 에만 있다.
  D5  infer.py:152  역정규화가 numpy 라 gradient 가 끊긴다. torch 로 다시 쓴다.

학습 대상은 ConvVAE decoder 뿐이다 (627,075 params). UNet 은 49.86M (97%) 이고
동결한 뒤 anatomy feature [1,512] 만 캐시하면 3D UNet 이 학습 그래프에서 사라진다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[3]
# upstream 이 두 군데 중 하나에 풀려 있을 수 있다.
UPSTREAM = next((p for p in (_ROOT / "stable" / "stable",
                             _ROOT / "external" / "atm_upstream" / "stable")
                 if (p / "model" / "model.py").exists()),
                _ROOT / "external" / "atm_upstream" / "stable")

BUNDLES = [
    "AF_L", "AF_R", "CC_Fr_1", "CC_Fr_2", "CC_Oc", "CC_Pa", "CC_Pr_Po", "CG_L", "CG_R",
    "FAT_L", "FAT_R", "FPT_L", "FPT_R", "IFOF_L", "IFOF_R", "ILF_L", "ILF_R", "MCP",
    "MdLF_L", "MdLF_R", "OR_ML_L", "OR_ML_R", "POPT_L", "POPT_R", "PYT_L", "PYT_R",
    "SLF_L", "SLF_R", "UF_L", "UF_R",
]
ANATOMICAL_DIM, LATENT_DIM, N_POINTS = 512, 64, 128


def _import_upstream():
    p = str(UPSTREAM)
    if p not in sys.path:
        sys.path.insert(0, p)
    from model.model import ATMVAE                                # noqa: E402
    return ATMVAE


class BundleNorm:
    """bundle 별 정규화 상수.

    upstream 값은 rigid-2009c 공간 기준이다. 우리 GT 는 QSDR/NLin6 공간이므로
    fine-tuning 시에는 우리 데이터에서 다시 계산한 상수를 써야 한다 (from_npz).
    """

    def __init__(self, t1_min, t1_max, coord_min, coord_max):
        self.t1_min, self.t1_max = float(t1_min), float(t1_max)
        self.coord_min = np.asarray(coord_min, np.float64).reshape(3)
        self.coord_max = np.asarray(coord_max, np.float64).reshape(3)
        assert self.t1_max > self.t1_min, (self.t1_min, self.t1_max)
        assert (self.coord_max > self.coord_min).all(), (self.coord_min, self.coord_max)

    @classmethod
    def from_upstream(cls, bundle: str) -> "BundleNorm":
        s = UPSTREAM / "supp"                      # D4: data/ 가 아니라 supp/
        g = lambda n: np.load(s / f"{bundle}_{n}.npy")
        return cls(g("min_vals_rigid_transformed_T1w"), g("max_vals_rigid_transformed_T1w"),
                   g("min_coords_rigid_transformed_streamlines"),
                   g("max_coords_rigid_transformed_streamlines"))

    @classmethod
    def from_npz(cls, path) -> "BundleNorm":
        z = np.load(path)
        return cls(z["t1_min"], z["t1_max"], z["coord_min"], z["coord_max"])

    def normalize_t1(self, vol: np.ndarray, robust: bool = True, target: float = 0.6,
                     pct: float = 99.5, unit: bool = False) -> np.ndarray:
        """upstream 식: (x - min) / (max - min).

        robust=True (기본): 그 전에 subject 별로 뇌 안(>0) 강도의 pct 백분위가 정규화 후
        `target` 이 되도록 스케일을 맞춘다. PPMI T1 은 native max 가 878 ~ 203,163 으로
        subject 마다 230배까지 달라(스캐너/프로토콜) upstream 의 고정 상수(8330)만으로는
        정규화 후 >1 인 voxel 이 39 % 인 subject 가 생기고 anatomy feature 가 40배 커진다.
        target=0.6 은 sub-000001(정규화 후 max 0.66) 이 있던 영역이다.

        unit=True: 결과를 [0,1] 로 보장한다 (전처리 프로토콜). robust 스케일링만으로는
        상위 0.5 % 가 1 을 넘는다 (실측 max 1.208, 1.237). pct 백분위를 1.0 에 맞추고
        그 위를 자른다 -- 밝은 이상치(지방/혈관)가 스케일을 지배하지 않게 하면서
        범위는 확실히 [0,1] 이 된다. **모델 입력이 바뀌므로 재학습이 필요하다.**
        """
        if unit:
            fg = vol[vol > 0]
            assert fg.size > 1000, "T1 이 거의 비어 있음"
            scale = float(np.percentile(fg, pct))
            assert scale > 0, "T1 백분위가 0"
            out = np.clip(vol / scale, 0.0, 1.0).astype(np.float32)
            assert 0.0 <= out.min() and out.max() <= 1.0, (out.min(), out.max())
            return out
        if robust:
            fg = vol[vol > 0]
            assert fg.size > 1000, "T1 이 거의 비어 있음"
            scale = float(np.percentile(fg, pct))
            assert scale > 0, "T1 백분위가 0"
            vol = vol / scale * (target * (self.t1_max - self.t1_min) + self.t1_min)
        return (vol - self.t1_min) / (self.t1_max - self.t1_min + 1e-6)


def _expand_conv_in_channels(unet, in_ch: int) -> None:
    """rigid_UNet 의 첫 conv 를 in_channels=1 -> in_ch 로 넓힌다 (WM 채널 도입, 재학습 설계 §2 ①).

    새 채널 가중치를 **0 으로** 초기화하므로 시작 시점 출력은 사전학습 1채널과 정확히 같다.
    (ch0 가 기존 T1 경로, ch1 이상은 기여 0 -> WM 경로만 새로 학습된다.)
    """
    old = unet.conv1_1
    assert old.in_channels == 1, f"이미 확장된 conv 다: in_channels={old.in_channels}"
    assert in_ch > 1, in_ch
    new = torch.nn.Conv3d(in_ch, old.out_channels, kernel_size=old.kernel_size, stride=old.stride,
                          padding=old.padding, dilation=old.dilation, groups=old.groups,
                          bias=old.bias is not None)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :1].copy_(old.weight)
        if old.bias is not None:
            new.bias.copy_(old.bias)
    assert float(new.weight[:, 1:].abs().max()) == 0.0, "새 채널이 0 이 아니다"
    unet.conv1_1 = new


class ATMBundle(torch.nn.Module):
    """bundle 하나에 대한 ATM 모델 + 미분 가능한 mm 출력."""

    def __init__(self, bundle: str, norm: BundleNorm | None = None,
                 device="cuda", train_mode_unet: bool = False,
                 models_dir: Path | None = None, in_channels: int = 1):
        """in_channels: UNet 입력 채널 수. 1 = upstream 그대로 (T1 만), 2 = T1 + WM 확률.
        upstream 재현 경로(scripts/00, 11)는 1 을 쓴다. 파이프라인 기본은 ROIPairATM 쪽이 2 다."""
        super().__init__()
        ATMVAE = _import_upstream()
        self.bundle = bundle
        self.norm = norm or BundleNorm.from_upstream(bundle)
        self.device = torch.device(device)

        mdir = Path(models_dir) if models_dir else UPSTREAM / "models"
        ckpt = mdir / bundle / f"atmvae_{bundle}.pth"
        assert ckpt.exists(), f"체크포인트 없음: {ckpt}"
        net = ATMVAE(ANATOMICAL_DIM, LATENT_DIM, True)
        sd = torch.load(ckpt, map_location="cpu", weights_only=True)
        missing, unexpected = net.load_state_dict(sd, strict=False)
        assert not missing and not unexpected, (missing, unexpected)
        self.in_channels = int(in_channels)
        if self.in_channels != 1:
            _expand_conv_in_channels(net.unet, self.in_channels)
        self.net = net.to(self.device)

        self.net.eval()                                   # D2
        self.train_mode_unet = train_mode_unet
        if train_mode_unet:
            self.net.unet.train()

        cmin = torch.tensor(self.norm.coord_min, dtype=torch.float32, device=self.device)
        cmax = torch.tensor(self.norm.coord_max, dtype=torch.float32, device=self.device)
        self.register_buffer("coord_min", cmin)
        self.register_buffer("coord_scale", (cmax - cmin) / 2.0)

    # --- 파라미터 그룹 -------------------------------------------------------
    def freeze_unet(self) -> None:
        for p in self.net.unet.parameters():
            p.requires_grad_(False)

    def decoder_parameters(self):
        names = ("decoder_fc", "deconv1", "deconv2", "deconv3", "dfilm1", "dfilm2", "dbn1", "dbn2")
        return [p for n, p in self.net.ae.named_parameters() if n.split(".")[0] in names]

    # --- forward -------------------------------------------------------------
    def _check_input(self, x: torch.Tensor, full_grid: bool = True) -> None:
        """UNet 입력 [1, in_channels, 193, 229, 193]. 1채널 볼륨을 2채널 모델에 넣으면
        conv 가 그냥 죽는 대신 여기서 이유가 보이게 멈춘다 (ch1 = WM 확률).

        full_grid=False 는 채널만 본다 -- 인코더는 fully-conv + global pool 이라 격자 크기에
        무관하고, 단위 테스트는 작은 격자를 쓴다.
        """
        assert x.ndim == 5 and tuple(x.shape[:2]) == (1, self.in_channels), (
            f"UNet 입력 shape {tuple(x.shape)} != (1, {self.in_channels}, ...)")
        if full_grid:
            assert tuple(x.shape[2:]) == (193, 229, 193), (
                f"UNet 입력 격자 {tuple(x.shape[2:])} != (193, 229, 193)")

    @staticmethod
    def _unet_encoder_only(u, x: torch.Tensor) -> torch.Tensor:
        """rigid_UNet 의 인코더 가지만 실행한다.

        model.py:193-224 와 완전히 같은 연산이다. 다르게 하는 것은 segmentation 을
        만드는 디코더 가지(model.py:227-248)를 아예 실행하지 않는 것뿐이고, 그 가지는
        anatomical_condition 에 영향을 주지 않는다 (infer.py:139 도 결과를 버린다).

        전체 forward 는 A10 23GB 에서 OOM 난다: 디코더가 193x229x193 에서 128채널
        concat(4.4GB)까지 만들기 때문이다. 인코더만 돌리면 peak 가 1/3 이하로 준다.
        수치적으로 동일하다는 것은 CPU 전체 forward 와 대조해 확인한다
        (scripts/00_check_env.py).
        """
        import torch.nn.functional as F
        assert x.shape[1] == u.conv1_1.in_channels, (x.shape, u.conv1_1.in_channels)
        c11 = F.relu(u.conv1_1(x)); c12 = F.relu(u.conv1_2(c11))
        o1 = c11 + F.relu(u.conv1_3(u.dropout1(c12)))
        del c11, c12
        c21 = F.relu(u.conv2_1(o1)); del o1
        o2 = c21 + F.relu(u.conv2_3(u.dropout2(F.relu(u.conv2_2(c21))))); del c21
        c31 = F.relu(u.conv3_1(o2)); del o2
        o3 = c31 + F.relu(u.conv3_3(u.dropout3(F.relu(u.conv3_2(c31))))); del c31
        c41 = F.relu(u.conv4_1(o3)); del o3
        o4 = c41 + F.relu(u.conv4_3(u.dropout4(F.relu(u.conv4_2(c41))))); del c41
        h = u.global_avg_pool(o4).view(o4.size(0), -1)
        return u.fc(h)

    # --- rigid_UNet 인코더를 stage 로 나눈다 (model.py:193-224 와 연산 동일) -------------
    @staticmethod
    def _stage1(u, x):
        import torch.nn.functional as F
        c11 = F.relu(u.conv1_1(x)); c12 = F.relu(u.conv1_2(c11))
        return c11 + F.relu(u.conv1_3(u.dropout1(c12)))

    @staticmethod
    def _stage2(u, o1):
        import torch.nn.functional as F
        c21 = F.relu(u.conv2_1(o1))
        return c21 + F.relu(u.conv2_3(u.dropout2(F.relu(u.conv2_2(c21)))))

    @staticmethod
    def _stage3(u, o2):
        import torch.nn.functional as F
        c31 = F.relu(u.conv3_1(o2))
        return c31 + F.relu(u.conv3_3(u.dropout3(F.relu(u.conv3_2(c31)))))

    _STAGES = ("stage1", "stage2", "stage3", "stage4")
    UNET_LEVELS = ("none", "stage4", "stage3", "stage2", "full")   # 'full' == stage1 부터 전부

    def unet_stage_parameters(self, stage: str):
        u = self.net.unet
        mods = {"stage1": (u.conv1_1, u.conv1_2, u.conv1_3),
                "stage2": (u.conv2_1, u.conv2_2, u.conv2_3),
                "stage3": (u.conv3_1, u.conv3_2, u.conv3_3),
                "stage4": (u.conv4_1, u.conv4_2, u.conv4_3, u.fc)}[stage]
        return [p for m in mods for p in m.parameters()]

    def trainable_unet_stages(self, level: str):
        assert level in self.UNET_LEVELS, level
        if level == "none":
            return ()
        first = {"stage4": 3, "stage3": 2, "stage2": 1, "full": 0}[level]
        return self._STAGES[first:]

    def set_unet_trainable(self, level: str) -> None:
        """T1 encoder unfreeze. level 이상의 stage 만 requires_grad. 디코더 가지(segmentation)는
        절대 학습하지 않는다 (사용하지 않는 경로)."""
        self.freeze_unet()
        for st in self.trainable_unet_stages(level):
            for p in self.unet_stage_parameters(st):
                p.requires_grad_(True)
        self.unet_level = level

    def unet_trainable_parameters(self):
        return [p for st in self.trainable_unet_stages(getattr(self, "unet_level", "none"))
                for p in self.unet_stage_parameters(st)]

    def encode_anatomy_grad(self, t1_w: torch.Tensor, level: str | None = None,
                            use_checkpoint: bool = True, return_stage3: bool = False):
        """T1 -> anatomy feature, level 이상의 stage 에 gradient 가 흐르는 경로.

        동결된 앞부분은 no_grad 로 돌리고, 학습하는 stage 는 torch.utils.checkpoint 로 감싸
        forward 에서 중간 활성을 저장하지 않는다. stage1 은 193x229x193 에서 64ch 텐서가
        4~5 개(각 2.2 GB) 라 checkpoint 없이는 23 GB GPU 에서 backward 가 불가능하다.
        subject 당 step 마다 **1회** 만 호출한다 (trainer 가 leaf 로 분리해 backward 도 1회).
        """
        from torch.utils.checkpoint import checkpoint
        level = level or getattr(self, "unet_level", "none")
        assert level != "none", "동결 인코더는 encode_anatomy() 를 쓴다"
        self._check_input(t1_w, full_grid=False)
        u = self.net.unet
        stages = [("stage1", lambda a: self._stage1(u, a)), ("stage2", lambda a: self._stage2(u, a)),
                  ("stage3", lambda a: self._stage3(u, a)), ("stage4", lambda a: self._unet_stage4_fc(u, a))]
        train = set(self.trainable_unet_stages(level))
        h = t1_w.to(self.device)
        o3 = None
        for name, fn in stages:
            if name not in train:
                with torch.no_grad():
                    h = fn(h)
            elif use_checkpoint:
                h = checkpoint(fn, h, use_reentrant=False)
            else:
                h = fn(h)
            if name == "stage3":
                # 인코더를 학습하면 디스크 캐시된 ROI 국소 feature 는 **낡은 값**이 된다.
                # 조용히 틀리는 종류라 살아있는 stage3 를 내보내 호출부에서 풀링하게 한다.
                o3 = h
        assert h.shape == (1, ANATOMICAL_DIM), h.shape
        assert o3 is not None and o3.ndim == 5, 'stage3 출력을 못 잡았다'
        return (h, o3) if return_stage3 else h

    @classmethod
    def _unet_stage3(cls, u, x):                       # 이전 이름 호환 (stage1~3)
        return cls._stage3(u, cls._stage2(u, cls._stage1(u, x)))

    @staticmethod
    def _unet_stage4_fc(u, o3: torch.Tensor) -> torch.Tensor:
        """conv4_x + global_avg_pool + fc (model.py:214-224). 25x29x25 x 512ch 라 학습 가능."""
        import torch.nn.functional as F
        c41 = F.relu(u.conv4_1(o3))
        o4 = c41 + F.relu(u.conv4_3(u.dropout4(F.relu(u.conv4_2(c41)))))
        return u.fc(u.global_avg_pool(o4).view(o4.size(0), -1))

    def stage4_parameters(self):
        u = self.net.unet
        return [p for m in (u.conv4_1, u.conv4_2, u.conv4_3, u.fc) for p in m.parameters()]

    def encode_anatomy_trainable(self, t1_w: torch.Tensor) -> torch.Tensor:
        """conv1~3 은 no_grad, conv4_x + fc 만 gradient. subject 판별력을 학습하기 위한 경로.

        동결된 pretrained feature 는 subject 간 cosine 0.97 로 거의 같아(실측) decoder 만으로는
        T1 조건이 subject 를 구분하지 못한다. stage3 출력 [1,256,49,58,49] 를 캐시하면
        (35 MB) 매 step 앞단을 다시 돌릴 필요도 없다 (cache_stage3 참고).
        """
        self._check_input(t1_w)
        with torch.no_grad():
            o3 = self._unet_stage3(self.net.unet, t1_w.to(self.device))
        return self._unet_stage4_fc(self.net.unet, o3)

    @torch.no_grad()
    def cache_stage3(self, t1_w: torch.Tensor) -> torch.Tensor:
        """[1,256,49,58,49] -- 학습 가능한 인코더 경로의 입력. subject 당 1회 계산해 저장."""
        self._check_input(t1_w)
        return self._unet_stage3(self.net.unet, t1_w.to(self.device))

    def anatomy_from_stage3(self, o3: torch.Tensor) -> torch.Tensor:
        """캐시된 stage3 출력 -> anatomy feature (conv4_x + fc 를 통과, gradient 있음)."""
        return self._unet_stage4_fc(self.net.unet, o3.to(self.device))

    @torch.no_grad()
    def encode_anatomy(self, t1_w: torch.Tensor, full: bool = False) -> torch.Tensor:
        """T1 [1,1,193,229,193] -> anatomy feature [1,512]. subject x bundle 당 1회.

        full=True 면 upstream 그대로 (segmentation 포함) 실행한다. 23GB GPU 에서는 OOM.
        """
        self._check_input(t1_w)
        x = t1_w.to(self.device)
        a = self.net.unet(x)[0] if full else self._unet_encoder_only(self.net.unet, x)
        assert a.shape == (1, ANATOMICAL_DIM) and torch.isfinite(a).all()
        assert a.abs().max() > 0, "anatomy feature 가 전부 0 — T1 정규화 확인"
        return a

    def decode_mm(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """z [N,64] + anatomy [1,512] 또는 [N,512] -> streamline [N,128,3] (mm).

        upstream 은 여기서 (a) 3000 을 하드코딩하고 (b) numpy 로 역정규화한다.
        둘 다 여기서 고친다 -- 역정규화는 affine 이므로 gradient 가 그대로 흐른다.
        """
        assert z.ndim == 2 and z.shape[1] == LATENT_DIM, z.shape
        n = z.shape[0]
        if a.shape[0] == 1:
            a = a.expand(n, -1)                            # D3
        assert a.shape == (n, ANATOMICAL_DIM), a.shape
        raw = self.decode_raw(z, a)
        mm = (raw + 1.0) * self.coord_scale + self.coord_min   # D5
        assert mm.shape == (n, N_POINTS, 3)
        return mm

    def decode_raw(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """정규화 전 디코더 출력 [N,128,3], tanh 라 [-1,1].

        좌표 역정규화를 분리한 이유: 전역 뇌 박스 하나로 3,321 쌍을 다 덮으면
        실측상 pair 실제 범위의 **28배 부피**(선형 3.0배)를 tanh 한 구간에 욱여넣게 된다
        (upstream 은 번들마다 supp/{bundle}_{min,max}_coords_*.npy 로 박스를 따로 뒀다).
        pair 앵커 재매개화(roi_atm.PairAnchor)가 이 출력을 받아 다른 상수로 되돌린다.
        """
        n = z.shape[0]
        if a.shape[0] == 1:
            a = a.expand(n, -1)
        s = self.net.ae.decode(z, a)                       # [N,3,128], tanh -> [-1,1]
        assert s.shape == (n, 3, N_POINTS), s.shape
        return s.permute(0, 2, 1)

    def encode_streamline(self, mm: torch.Tensor, a: torch.Tensor):
        """GT streamline [N,128,3] (mm) -> (mu, logvar). L_stream 용."""
        n = mm.shape[0]
        assert mm.shape[1:] == (N_POINTS, 3), mm.shape
        s = (mm - self.coord_min) / self.coord_scale - 1.0
        if a.shape[0] == 1:
            a = a.expand(n, -1)
        return self.net.ae.encode(s.permute(0, 2, 1), a)


    # --- GPU 규칙 (Framework v2 §19, 사용자 사양 §5) --------------------------
    def decode_chunks(self, z: torch.Tensor, a: torch.Tensor, chunk: int = 4096,
                      amp_dtype: torch.dtype | None = None):
        """latent batch 를 chunk 로 나눠 decode. streamline 마다 encoder 를 다시 부르지
        않는다 -- anatomy feature `a` 는 이미 계산된 것을 재사용한다.

        SC 는 streamline 에 대해 가산적이므로 호출측에서 chunk 별 부분 SC 를 더한 뒤
        마지막에 한 번만 loss 를 계산해야 한다 (ChunkedSC 참고).

        amp_dtype: torch.bfloat16 등. decoder 만 저정밀도로 돌리고 좌표는 fp32 로 되돌린다
                   (SC 의 log/상관/정규화는 fp32 여야 한다).
        """
        assert z.ndim == 2 and z.shape[1] == LATENT_DIM, z.shape
        for i in range(0, z.shape[0], chunk):
            zc = z[i:i + chunk]
            ac = a if a.shape[0] == 1 else a[i:i + chunk]
            if amp_dtype is None:
                yield self.decode_mm(zc, ac)
            else:
                with torch.autocast(device_type=self.device.type, dtype=amp_dtype):
                    mm = self.decode_mm(zc, ac)
                yield mm.float()

    @torch.inference_mode()
    def generate(self, n: int, a: torch.Tensor, chunk: int = 16384, seed: int | None = None,
                 amp_dtype: torch.dtype | None = None) -> torch.Tensor:
        """inference 전용. backward graph 가 없으므로 학습보다 큰 chunk 를 쓴다.
        좌표를 전부 GPU 에서 만든 뒤 마지막에 CPU 로 내린다 (.trk 저장은 CPU 작업)."""
        z = torch.from_numpy(sample_latents(self.bundle, n, seed=seed)).to(self.device)
        return torch.cat([c for c in self.decode_chunks(z, a, chunk, amp_dtype)])


def sample_latents(bundle: str, n: int, kde_dir: Path | None = None, seed: int | None = None):
    """upstream KDE 에서 latent 를 뽑는다 (tophat, bw=1, 447k 학습 샘플).

    ATM 의 latent prior 는 N(0, I) 가 아니라 학습 latent 은행이다. 임의 정규분포에서
    뽑으면 안 된다.
    """
    kde = load_kde(bundle, kde_dir)
    # sklearn 의 KernelDensity.sample 은 self.random_state 가 아니라 **인자**를 쓴다.
    # kde.random_state = seed 로는 재현되지 않는다 (같은 seed 로 max|z1-z2| = 7.7 확인).
    z = kde.sample(n, random_state=seed)
    assert z.shape == (n, LATENT_DIM) and np.isfinite(z).all()
    return z.astype(np.float32)


_KDE_CACHE: dict = {}


def load_kde(bundle: str, kde_dir: Path | None = None):
    """KDE 로드는 약 30초 걸리므로 캐시한다 (tophat, bw=1, 447,000 x 64 학습 샘플)."""
    d = Path(kde_dir) if kde_dir else UPSTREAM / "kde_models"
    p = d / bundle / "kde_model.joblib"
    assert p.exists(), f"KDE 없음: {p}"
    if p not in _KDE_CACHE:
        import joblib
        _KDE_CACHE[p] = joblib.load(p)
    return _KDE_CACHE[p]
