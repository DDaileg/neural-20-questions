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
    agent = BeliefAgent()
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

    Stateless — all state lives in BeliefState. Call update() to get a
    new BeliefState; the old one is unchanged.
    """

    def update(
        self,
        state: BeliefState,
        attr: str,
        answer: str,
    ) -> BeliefState:
        """
        Applies a yes/no answer to produce a new BeliefState.

        Args:
            state  : current BeliefState
            attr   : attribute that was asked
            answer : user's answer — "y" or "n"

        Returns:
            New BeliefState with filtered candidates, updated asked set,
            and incremented question count.
        """
        filtered = filter_candidates(state.candidates, attr, answer)
        new_asked = state.asked | {attr}
        return BeliefState(
            candidates=filtered,
            asked=new_asked,
            question_count=state.question_count + 1,
        )
