"""
belief_agent.py
---------------
Agent responsible for tracking what the system currently believes.

BeliefState is a plain data object: the current candidate DataFrame, the
set of attributes already asked, and the count of questions used so far.

BeliefAgent owns the update logic — given an attribute and a yes/no answer,
it returns a new BeliefState. Immutable updates keep the game loop clean.

When Phase 2 lands (spaCy embeddings), BeliefAgent.update() is where boolean
filtering gets replaced with cosine similarity scoring — nothing else changes.

Interface:
    state = BeliefState(df)
    agent = BeliefAgent(vectors)
    state = agent.update(state, attr, answer)
    names = state.candidate_names
    count = state.candidate_count
"""

import pandas as pd
from candidate_filter import filter_candidates, filter_candidates_by_conceptnet, get_candidate_names


class BeliefState:
    """
    Immutable snapshot of what the system currently believes.

    Attributes:
        candidates       : DataFrame of objects still consistent with all answers
        asked            : set of attribute names already used this game
        question_count   : number of valid questions asked so far
    """

    def __init__(
        self,
        candidates: pd.DataFrame,
        asked: set | None = None,
        question_count: int = 0,
    ):
        self.candidates = candidates
        self.asked = asked if asked is not None else set()
        self.question_count = question_count

    @property
    def candidate_count(self) -> int:
        return len(self.candidates)

    @property
    def candidate_names(self) -> list[str]:
        return get_candidate_names(self.candidates)


class BeliefAgent:
    """
    Updates belief state after each answer.

    Args:
        vectors   : precomputed word vectors from data_loader
        cn_client : ConceptNetClient instance; if None, CN questions are not
                    supported and any "cn:..." key falls back to no-op filtering
    """

    def __init__(self, vectors: dict, cn_client=None):
        self.vectors = vectors
        self.cn = cn_client

    def update(
        self,
        state: BeliefState,
        attr: str,
        answer: str,
    ) -> BeliefState:
        """
        Applies a yes/no answer to produce a new BeliefState.

        Routes ConceptNet question keys ("cn:...") to
        filter_candidates_by_conceptnet(); all other keys use the
        boolean + cosine filter_candidates() path.
        """
        if attr.startswith("cn:") and self.cn is not None:
            filtered = filter_candidates_by_conceptnet(
                state.candidates, attr, answer, self.cn, self.vectors
            )
        else:
            filtered = filter_candidates(state.candidates, attr, answer, self.vectors)

        new_asked = state.asked | {attr}
        return BeliefState(
            candidates=filtered,
            asked=new_asked,
            question_count=state.question_count + 1,
        )
