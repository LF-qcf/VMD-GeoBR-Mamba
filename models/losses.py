import torch
import torch.nn.functional as F

TV_RATIO = 0.05
GATE_RATIO = 2.5


def charbonnier(x, eps=1e-6):
    """Charbonnier penalty: smooth approximation to L1 loss."""
    return torch.sqrt(x * x + eps)


def second_order_smoothness(bg):
    if bg.shape[1] < 3:
        return bg.sum() * 0.0
    d2 = bg[:, 2:] - 2.0 * bg[:, 1:-1] + bg[:, :-2]
    return torch.mean(torch.abs(d2))


def first_order_tv(bg):
    if bg.shape[1] < 2:
        return bg.sum() * 0.0
    d1 = bg[:, 1:] - bg[:, :-1]
    return torch.mean(torch.abs(d1))


def geobr_decomposition_objective(out, x_seq, y_future, args):
    x_seq = x_seq.squeeze(-1)

    bg_seq = out["bg_seq"]
    res_seq = out["res_seq"]
    bg_future = out["bg_future"]
    res_future = out["res_future"]
    seq_gate = out["seq_gate"]
    future_gate = out["future_gate"]

    recon_seq = bg_seq + res_seq
    recon_future = bg_future + res_future

    if args.recon_penalty == "mse":
        obs_seq = F.mse_loss(recon_seq, x_seq)
        obs_future = F.mse_loss(recon_future, y_future)
    else:
        obs_seq = torch.mean(charbonnier(recon_seq - x_seq, eps=args.charbonnier_eps))
        obs_future = torch.mean(charbonnier(recon_future - y_future, eps=args.charbonnier_eps))

    data_consistency = 0.5 * (obs_seq + obs_future)

    bg_all = torch.cat([bg_seq, bg_future], dim=1)
    bg_second = second_order_smoothness(bg_all)
    bg_first = first_order_tv(bg_all)
    background_prior = bg_second + TV_RATIO * bg_first

    residual_magnitude = torch.mean(torch.abs(res_seq)) + torch.mean(torch.abs(res_future))
    gate_activation = torch.mean(seq_gate) + torch.mean(future_gate)
    residual_prior = residual_magnitude + GATE_RATIO * gate_activation

    total = data_consistency + args.lambda_bg * background_prior + args.lambda_res * residual_prior

    items = {
        "total": float(total.detach().cpu()),
        "data_consistency": float(data_consistency.detach().cpu()),
        "background_prior": float(background_prior.detach().cpu()),
        "residual_prior": float(residual_prior.detach().cpu()),
        "obs_seq": float(obs_seq.detach().cpu()),
        "obs_future": float(obs_future.detach().cpu()),
        "bg_second": float(bg_second.detach().cpu()),
        "bg_first": float(bg_first.detach().cpu()),
        "residual_magnitude": float(residual_magnitude.detach().cpu()),
        "gate_activation": float(gate_activation.detach().cpu()),
    }
    return total, items
