import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def parse_kernels(text):
    if isinstance(text, (tuple, list)):
        return tuple(int(x) for x in text)

    kernels = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            kernels.append(int(item))

    if not kernels:
        raise ValueError("--kernels cannot be empty, for example: --kernels 3,7,15")

    return tuple(kernels)


def resolve_project_path(path, project_root):
    path = Path(path)
    if path.is_absolute():
        return str(path)
    return str(Path(project_root) / path)
