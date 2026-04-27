"""
candidate_filter.py
-------------------
Responsible for narrowing the candidate set based on a yes/no answer.
No entropy logic, no I/O, no data loading — pure filtering.

When Phase 2 lands (spaCy embeddings), this is where soft similarity
filtering replaces exact boolean matching — without touching anything else.
"""

import pandas as pd


def filter_candidates(
    df_subset: pd.DataFrame,
    attr: str,
    answer: str,
) -> pd.DataFrame:
    """
    Filters the candidate DataFrame to only rows that match the user's answer.

    Args:
        df_subset : DataFrame of current candidate objects
        attr      : attribute column name to filter on (e.g. "is_alive")
        answer    : raw user input — "y" or "n"

    Returns:
        Filtered DataFrame containing only objects consistent with the answer.
    """
    expected = (answer == "y")
    return df_subset[df_subset[attr] == expected].reset_index(drop=True)


def get_candidate_names(df_subset: pd.DataFrame) -> list[str]:
    """
    Returns the list of candidate object names from the current subset.
    Used by the game loop and reasoning trace for display purposes.
    """
    return df_subset["Name"].tolist()
