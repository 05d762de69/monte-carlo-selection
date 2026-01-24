from __future__ import annotations

from typing import List

import numpy as np


def component_members(
    rows_xy: List[np.ndarray],
    resp: np.ndarray,
    *,
    k: int,
    top_n: int = 250,
) -> List[np.ndarray]:
    """
    Return the top-N members for component k by responsibility.
    """
    r = resp[:, k]
    idx = np.argsort(-r)[: int(top_n)]
    return [rows_xy[int(i)] for i in idx]
