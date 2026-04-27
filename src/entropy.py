"""
entropy.py
----------
Responsible for question selection using Shannon binary entropy.
No data loading, no game logic, no I/O — pure information theory.

When Phase 2 lands (spaCy embeddings), the selector can be swapped out
here without touching anything else in the system.
"""

import numpy as np
import pandas as pd


def binary_entropy(p: float) -> float:
    """
    Shannon binary entropy for a probability p.

    Returns 0.0 at certainty (p=0 or p=1) — the question tells you nothing.
    Returns 1.0 at p=0.5 — the question splits candidates perfectly in half.

    H(p) = -p·log₂(p) - (1-p)·log₂(1-p)
    """
    if p <= 0 or p >= 1:
        return 0.0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def choose_best_question(
    df_subset: pd.DataFrame,
    attributes: list[str],
    asked: set,
) -> tuple[str | None, float]:
    """
    Selects the attribute with the highest entropy score from the current
    candidate set. High entropy = question splits remaining candidates most
    evenly = maximum information gained per question.

    Args:
        df_subset  : DataFrame of remaining candidate objects
        attributes : full list of attribute column names to consider
        asked      : set of attribute names already used this game

    Returns:
        (best_attr, entropy_score) — or (None, 0.0) if no questions remain
    """
    best_attr = None
    max_entropy = -1.0

    for attr in attributes:
        if attr in asked:
            continue
        p = df_subset[attr].mean()
        score = binary_entropy(p)
        if score > max_entropy:
            max_entropy = score
            best_attr = attr

    if best_attr is None:
        return None, 0.0

    return best_attr, max_entropy
