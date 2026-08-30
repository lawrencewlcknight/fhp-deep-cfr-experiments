"""Reproducibility helpers.

These helpers seed every global random-number generator that the rest of the
package consults: Python's :mod:`random`, :mod:`numpy.random`, PyTorch (CPU and
CUDA), and the ``PYTHONHASHSEED`` environment variable. They also pin cuDNN
into deterministic mode, which is a no-op on CPU-only hosts.

Note that mutating global RNG state is convenient but means two solvers running
in the same Python process share state. For multi-experiment isolation prefer
constructing dedicated :class:`numpy.random.Generator` and :class:`torch.Generator`
objects and threading them through the API.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seeds Python, NumPy, and PyTorch (CPU + CUDA) RNGs.

    Sets ``PYTHONHASHSEED`` for hash determinism in subprocesses started after
    the call, and pins cuDNN to deterministic / non-benchmark mode.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
