import numpy as np
import torch
from torch.utils.data import Dataset


def get_ckpt_value(ckpt, name, default=None):
    if name in ckpt:
        return ckpt[name]

    args = ckpt.get("args", {})
    if isinstance(args, dict):
        if name in args:
            return args[name]
        alt = name.replace("-", "_")
        if alt in args:
            return args[alt]

    return default


class IMF3InferDataset(Dataset):
    def __init__(self, values_norm, values_raw, dates, seq_len, pred_len, start=0, end=None):
        self.values_norm = np.asarray(values_norm, dtype=np.float32)
        self.values_raw = np.asarray(values_raw, dtype=np.float32)
        self.dates = np.asarray(dates)
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.starts = []

        if end is None:
            end = len(self.values_norm)

        last_start = int(end) - self.seq_len - self.pred_len
        for idx in range(int(start), last_start + 1):
            self.starts.append(idx)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, item):
        s = self.starts[item]
        x_begin = s
        x_end = s + self.seq_len
        y_begin = x_end
        y_end = y_begin + self.pred_len

        x_norm = self.values_norm[x_begin:x_end, None]
        y_norm = self.values_norm[y_begin:y_end]
        y_raw = self.values_raw[y_begin:y_end]

        return (
            torch.from_numpy(x_norm).float(),
            torch.from_numpy(y_norm).float(),
            torch.from_numpy(y_raw).float(),
            torch.tensor(s, dtype=torch.long),
        )


@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()

    starts = []
    pred_bg_norm = []
    pred_res_norm = []
    pred_recon_norm = []
    raw_future_norm = []
    raw_future = []
    gates = []

    for xb, yb, y_raw, sb in loader:
        xb = xb.to(device, non_blocking=True)
        out = model(xb)

        bg_f = out["bg_future"].detach().cpu().numpy()
        res_f = out["res_future"].detach().cpu().numpy()
        recon_f = bg_f + res_f

        pred_bg_norm.append(bg_f)
        pred_res_norm.append(res_f)
        pred_recon_norm.append(recon_f)
        raw_future_norm.append(yb.numpy())
        raw_future.append(y_raw.numpy())
        gates.append(out["future_gate"].detach().cpu().numpy())
        starts.extend(sb.numpy().tolist())

    return (
        np.concatenate(pred_bg_norm, axis=0),
        np.concatenate(pred_res_norm, axis=0),
        np.concatenate(pred_recon_norm, axis=0),
        np.concatenate(raw_future_norm, axis=0),
        np.concatenate(raw_future, axis=0),
        np.concatenate(gates, axis=0),
        np.asarray(starts),
    )
