import torch
import torch.nn as nn
import torch.nn.functional as F
from agent.QCNN.Embedder import Embedder  # kept for compatibility with your project structure
from einops.layers.torch import Rearrange


def _make_dropout(p):
    """Return nn.Dropout(p) if p>0 else Identity (also works for p=None)."""
    if p is None:
        return nn.Identity()
    try:
        p = float(p)
    except Exception:
        return nn.Identity()
    return nn.Dropout(p) if p > 0 else nn.Identity()


class LayerNorm1d(nn.Module):
    """LayerNorm over channels for each position. Input/Output: [B, C, L]."""
    def __init__(self, channels: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.ln = nn.LayerNorm(channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, C, L] -> [B, L, C] -> LN(C) -> [B, C, L]
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


class MaskedDepthwiseConv1d(nn.Module):
    """
    Depthwise 1D conv with 'partial-conv' style renormalization to reduce padding-length sensitivity.

    x:    [B, C, L]
    mask: [B, 1, L]   (1 = valid item, 0 = pad)
    """
    def __init__(self, channels: int, kernel_size: int, padding: int, eps: float = 1e-6):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.padding = int(padding)
        self.eps = float(eps)

        self.conv = nn.Conv1d(
            channels, channels,
            kernel_size=self.kernel_size,
            padding=self.padding,
            groups=channels,
            bias=False,
        )

        # For denom = sum(mask) inside each conv window
        self.register_buffer("ones_kernel", torch.ones(1, 1, self.kernel_size))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # enforce padded positions are 0 BEFORE conv
        x = x * mask

        out = self.conv(x)  # [B, C, L]

        # denom: [B, 1, L] = number of valid items in each window
        denom = F.conv1d(mask, self.ones_kernel, padding=self.padding)  # [B, 1, L]

        # partial-conv renorm: scale up when fewer valid elements exist
        scale = (self.kernel_size / (denom + self.eps)).clamp(max=float(self.kernel_size))
        out = out * scale  # broadcast [B, C, L] * [B, 1, L]

        # keep padded positions dead
        out = out * mask
        return out


class ResNet1DBlock(nn.Module):
    """
    Mask-safe ResNet-like block for sequences of items.
    - Spatial mixing: masked depthwise conv (kernel k)
    - Channel mixing: pointwise conv (1x1)
    - Norm: LayerNorm per position (does not mix across L)
    - Mask is applied after every major step to prevent pad leakage.

    Input/Output: h [B, C, L], mask [B, 1, L]
    """
    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.0):
        super().__init__()
        k = int(kernel_size)
        pad = k // 2
        drop = _make_dropout(dropout)

        # (dw k) -> LN -> ReLU -> drop -> (pw 1) -> LN -> ReLU -> drop
        self.dw1 = MaskedDepthwiseConv1d(channels, kernel_size=k, padding=pad)
        self.ln1 = LayerNorm1d(channels)
        self.act = nn.ReLU()
        self.drop1 = drop
        self.pw1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.ln2 = LayerNorm1d(channels)
        self.drop2 = drop

        # (dw k) -> LN -> ReLU -> drop -> (pw 1) -> LN
        self.dw2 = MaskedDepthwiseConv1d(channels, kernel_size=k, padding=pad)
        self.ln3 = LayerNorm1d(channels)
        self.drop3 = drop
        self.pw2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.ln4 = LayerNorm1d(channels)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # ensure dead pads at entry
        h = h * mask
        residual = h

        out = self.dw1(h, mask)
        out = self.ln1(out)
        out = out * mask
        out = self.act(out)
        out = self.drop1(out)
        out = out * mask

        out = self.pw1(out)
        out = self.ln2(out)
        out = out * mask
        out = self.act(out)
        out = self.drop2(out)
        out = out * mask

        out = self.dw2(out, mask)
        out = self.ln3(out)
        out = out * mask
        out = self.act(out)
        out = self.drop3(out)
        out = out * mask

        out = self.pw2(out)
        out = self.ln4(out)
        out = out * mask

        out = out + residual
        out = self.act(out)
        out = out * mask
        return out


class QCNN(nn.Module):
    """
    Updated QCNN:
    - Replace old CNN backbone with a mask-safe ResNet1D backbone.
    - Keep init params the same, plus add n_res_blocks.
    - Dropout is disabled automatically if dropout<=0 or dropout=None.

    Inputs:
      x: [B, 10]
      y: [B, max_items, 23]
      z: [B, max_items]  (1 = real item, 0 = pad)
    """
    def __init__(self, d_model: int = 48, n_actions: int = 50, d_hidden: int = 64, dropout: float = 0.1, n_res_blocks: int = 6):
        super().__init__()
        self.d_model = d_model
        self.n_actions = n_actions
        self.d_hidden = d_hidden
        self.n_res_blocks = n_res_blocks

        drop = _make_dropout(dropout)

        # -------- item embedding (per-item) --------
        self.item_pre = nn.Sequential(
            nn.Linear(23, d_model),
            nn.ReLU(),
            drop,
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            Rearrange("b n d -> b d n"),  # -> [B, d_model, L]
        )

        # -------- ResNet1D backbone --------
        # Default kernel schedule that matches what you discussed:
        #   first 2 blocks use k=5, remaining blocks use k=3
        kernels = [5] * min(2, n_res_blocks) + [3] * max(0, n_res_blocks - 2)
        self.item_blocks = nn.ModuleList([
            ResNet1DBlock(d_model, kernel_size=k, dropout=dropout) for k in kernels
        ])

        # -------- env branch --------
        self.env_extractor = nn.Sequential(
            nn.Linear(10, d_model),
            nn.ReLU(),
            drop,
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )

        # -------- head --------
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.ReLU(),
            drop,
            nn.Linear(d_hidden, n_actions),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # 1) env part
        env_emb = self.env_extractor(x)  # [B, d_model]

        # 2) build mask and (optionally) zero-out padded raw inputs
        mask = z.unsqueeze(1).to(dtype=y.dtype)          # [B, 1, L]
        y = y * z.unsqueeze(-1).to(dtype=y.dtype)        # [B, L, 23] (hard zero pad input)

        # 3) item embedding + resnet1d backbone (mask-safe)
        h = self.item_pre(y)                              # [B, d_model, L]
        h = h * mask                                      # kill bias-leak from MLP/LN

        for blk in self.item_blocks:
            h = blk(h, mask)

        # 4) masked mean pooling over L
        h_masked = h * mask
        denom = mask.sum(dim=-1).clamp(min=1.0)           # [B, 1]
        item_emb = h_masked.sum(dim=-1) / denom           # [B, d_model]

        # 5) combine + predict Q
        combined = item_emb + env_emb                     # [B, d_model]
        q_values = self.predictor(combined)               # [B, n_actions]
        return q_values
