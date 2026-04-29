"""
guesser_agent.py
----------------
Agent responsible for deciding when to guess and what to guess.

Owns the stopping condition and the final output decision. GameRunner asks
this agent "should we stop?" and "what's your guess?" — it doesn't make
either decision itself.

When Phase 3 lands (neural question selector + confidence scoring), this
is where confidence thresholds and probabilistic guessing live. Right now
the logic is deterministic: stop when one candidate remains.

Interface:
    agent = GuesserAgent()
    if agent.should_guess(belief_state):
        result = agent.guess(belief_state)
        # result.guessed, result.name, result.alternatives
"""

from dataclasses import dataclass
from belief_agent import BeliefState


@dataclass
class GuessResult:
    """
    The outcome of a guess attempt.

    Attributes:
        guessed      : True if the system identified a single answer
        name         : the guessed object name (if guessed=True)
        alternatives : list of remaining candidates (if guessed=False and count > 0)
        exhausted    : True if no candidates remain (contradictory answers)
    """
    guessed: bool
    name: str | None
    alternatives: list[str]
    exhausted: bool


class GuesserAgent:
    """
    Decides when to stop asking and what the final answer is.

    Currently: guess when exactly one candidate remains.
    Phase 3: guess when model confidence exceeds a threshold.
    """

    def should_guess(self, state: BeliefState) -> bool:
        """
        Returns True when the game loop should stop and a guess should be made.
        Triggers on: single candidate, no candidates, or no attributes left.
        """
        return state.candidate_count <= 1

    def guess(self, state: BeliefState) -> GuessResult:
        """
        Produces the final guess from the current belief state.

        Returns:
            GuessResult with outcome details.
        """
        if state.candidate_count == 1:
            return GuessResult(
                guessed=True,
                name=state.candidate_names[0],
                alternatives=[],
                exhausted=False,
            )
        elif state.candidate_count == 0:
            return GuessResult(
                guessed=False,
                name=None,
                alternatives=[],
                exhausted=True,
            )
        else:
            # Fallback: multiple candidates remain (called at question limit)
            return GuessResult(
                guessed=False,
                name=None,
                alternatives=state.candidate_names,
                exhausted=False,
            )
