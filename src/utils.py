"""Shared helpers: config loading, seeding, paths."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path = ROOT / "configs" / "default.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Fix every random source we use so results are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def resolve(path: str | Path) -> Path:
    """Paths in the config are relative to the repo root."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p
