"""
belief_agent.py
---------------
Agent responsible for tracking what the system currently believes.

BeliefState is a plain data object: the current candidate DataFrame, the
set of attributes already asked, and the count of questions used so far.

BeliefAgent owns the update logic — given an attribute and a yes/no answer,
it returns a new BeliefState. Immutable updates keep the game loop clean.

update() always receives a static attribute column name — QuestionAgent
resolves any ConceptNet question to its mapped static attr before returning.

Interface:
    state = BeliefState(df)
    agent = BeliefAgent(vectors)
    state = agent.update(state, attr, answer)
    names = state.candidate_names
    count = state.candidate_count
"""

import pandas as pd
from candidate_filter import filter_candidates, get_candidate_names


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
        vectors : precomputed word vectors from data_loader
    """

    def __init__(self, vectors: dict):
        self.vectors = vectors

    def update(
        self,
        state: BeliefState,
        attr: str,
        answer: str,
    ) -> BeliefState:
        """
        Applies a yes/no answer to produce a new BeliefState.

        attr is always a static attribute column name — QuestionAgent
        resolves CN questions to their mapped static attr before this is called.
        """
        filtered = filter_candidates(state.candidates, attr, answer, self.vectors)

        new_asked = state.asked | {attr}
        return BeliefState(
            candidates=filtered,
            asked=new_asked,
            question_count=state.question_count + 1,
        )
