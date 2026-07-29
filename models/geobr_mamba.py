import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


class MultiScaleCausalConv(nn.Module):
    """Multi-scale causal convolution for IMF3 sequences."""

    def __init__(self, in_channels=1, d_model=64, kernels=(3, 7, 15), dropout=0.0):
        super().__init__()
        self.kernels = tuple(int(k) for k in kernels)
        self.branches = nn.ModuleList()

        for k in self.kernels:
            if k <= 0:
                raise ValueError(f"kernel size must be positive, got {k}")
            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=d_model,
                        kernel_size=k,
                        padding=0,
                        bias=False,
                    ),
                    nn.GELU(),
                )
            )

        self.fuse = nn.Sequential(
            nn.Conv1d(
                in_channels=d_model * len(self.kernels),
                out_channels=d_model,
                kernel_size=1,
                bias=False,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        outs = []
        for k, branch in zip(self.kernels, self.branches):
            x_pad = F.pad(x, (k - 1, 0))
            outs.append(branch(x_pad))
        y = torch.cat(outs, dim=1)
        y = self.fuse(y)
        return y.transpose(1, 2)


class ResidualShrinkBranch(nn.Module):
    """Learnable gated residual branch."""

    def __init__(self, d_model, pred_len=1, dropout=0.0):
        super().__init__()
        self.pred_len = int(pred_len)

        self.seq_res_raw = nn.Linear(d_model, 1)
        self.seq_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

        self.future_res_raw = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len),
        )
        self.future_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len),
            nn.Sigmoid(),
        )

    def forward(self, h_seq, h_last):
        seq_res_raw = self.seq_res_raw(h_seq).squeeze(-1)
        seq_gate = self.seq_gate(h_seq).squeeze(-1)
        seq_res = seq_gate * seq_res_raw

        future_res_raw = self.future_res_raw(h_last)
        future_gate = self.future_gate(h_last)
        future_res = future_gate * future_res_raw

        return seq_res, future_res, seq_gate, future_gate


class MambaSelfDecompBG(nn.Module):
    """Self-supervised background-residual decomposition network."""

    def __init__(
        self,
        d_model=64,
        pred_len=1,
        d_state=16,
        d_conv=4,
        expand=2,
        kernels=(3, 7, 15),
        dropout=0.0,
    ):
        super().__init__()
        self.pred_len = int(pred_len)

        self.local_encoder = MultiScaleCausalConv(
            in_channels=1,
            d_model=d_model,
            kernels=kernels,
            dropout=dropout,
        )

        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        self.norm_seq = nn.LayerNorm(d_model)
        self.norm_last = nn.LayerNorm(d_model)

        self.seq_bg_head = nn.Linear(d_model, 1)
        self.future_bg_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len),
        )

        self.residual_branch = ResidualShrinkBranch(
            d_model=d_model,
            pred_len=pred_len,
            dropout=dropout,
        )

    def forward(self, x):
        h = self.local_encoder(x)
        h = self.mamba(h)

        h_seq = self.norm_seq(h)
        h_last = self.norm_last(h[:, -1, :])

        bg_seq = self.seq_bg_head(h_seq).squeeze(-1)
        bg_future = self.future_bg_head(h_last)
        res_seq, res_future, seq_gate, future_gate = self.residual_branch(h_seq, h_last)

        return {
            "bg_seq": bg_seq,
            "res_seq": res_seq,
            "bg_future": bg_future,
            "res_future": res_future,
            "seq_gate": seq_gate,
            "future_gate": future_gate,
        }
