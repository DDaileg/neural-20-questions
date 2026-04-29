"""
candidate_filter.py
-------------------
Responsible for narrowing the candidate set based on yes/no answers.
No entropy logic, no I/O, no data loading — pure filtering and scoring.

Phase 2: filter_candidates() uses a hybrid approach (Option A corrected):
  1. Hard boolean filter on the answered attribute (keeps the game functional)
  2. Build a centroid from surviving candidates' vectors
  3. Score and rank survivors by cosine similarity to that centroid
  4. Scores are used for display in the reasoning trace — not for elimination

This means boolean filtering still drives narrowing (correctness), while
similarity scoring adds semantic ranking on top (portfolio signal).
"""

import numpy as np
import pandas as pd


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Returns cosine similarity between two vectors.
    Returns 0.0 if either vector is all zeros (OOV word).
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def filter_candidates(
    df_subset: pd.DataFrame,
    attr: str,
    answer: str,
    vectors: dict,
) -> pd.DataFrame:
    """
    Phase 2: Hybrid boolean + cosine similarity filtering.

    Step 1 — Hard boolean filter on the answered attribute.
              This is the primary narrowing mechanism. Keeps the game correct.

    Step 2 — Build a centroid from the surviving candidates' word vectors.

    Step 3 — Score each survivor by cosine similarity to that centroid.
              Scores are attached for display in the reasoning trace.
              They represent semantic cohesion among survivors — who "fits"
              most naturally with the remaining candidate cluster.

    Args:
        df_subset : DataFrame of current candidate objects
        attr      : attribute column name to filter on (e.g. "is_alive")
        answer    : raw user input — "y" or "n"
        vectors   : dict of {object_name: 300-dim numpy vector}

    Returns:
        Filtered DataFrame sorted by similarity score descending.
        Includes a "score" column for the reasoning trace.
    """
    expected = (answer == "y")

    # Step 1: hard boolean filter — the answer eliminates impossible candidates
    df_filtered = df_subset[df_subset[attr] == expected].reset_index(drop=True)

    if len(df_filtered) == 0:
        df_filtered = df_filtered.copy()
        df_filtered["score"] = 0.0
        return df_filtered

    # Step 2: build centroid from surviving candidates' vectors
    survivor_vectors = np.array([
        vectors[name] for name in df_filtered["Name"].tolist()
        if name in vectors and np.linalg.norm(vectors[name]) > 0
    ])

    if len(survivor_vectors) == 0:
        df_filtered = df_filtered.copy()
        df_filtered["score"] = 0.0
        return df_filtered

    centroid = survivor_vectors.mean(axis=0)

    # Step 3: score each survivor by similarity to the centroid
    scores = []
    for name in df_filtered["Name"].tolist():
        vec = vectors.get(name, np.zeros(300))
        scores.append(cosine_similarity(centroid, vec))

    df_filtered = df_filtered.copy()
    df_filtered["score"] = scores

    return df_filtered.sort_values("score", ascending=False).reset_index(drop=True)


def get_candidate_names(df_subset: pd.DataFrame) -> list[str]:
    """
    Returns the list of candidate object names from the current subset.
    Used by the game loop and reasoning trace for display purposes.
    """
    return df_subset["Name"].tolist()
