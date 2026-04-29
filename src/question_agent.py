"""
question_agent.py
-----------------
Agent responsible for selecting the next question to ask.

Owns the question selection strategy. Currently uses Shannon binary entropy
(greedy). When Phase 2 lands, swap the strategy here — BeliefAgent,
GuesserAgent, and GameRunner are unaffected.

Interface:
    agent = QuestionAgent(attributes, questions)
    attr, score, text = agent.select(belief_state)
"""

from entropy import choose_best_question


class QuestionAgent:
    """
    Selects the most informative question given the current belief state.

    Args:
        attributes : full list of attribute column names
        questions  : dict mapping attribute name -> natural language question
    """

    def __init__(self, attributes: list[str], questions: dict[str, str]):
        self.attributes = attributes
        self.questions = questions

    def select(self, belief_state: "BeliefState") -> tuple[str | None, float, str | None]:
        """
        Picks the highest-entropy unasked attribute from the current candidate set.

        Args:
            belief_state : current BeliefState instance

        Returns:
            (attr, entropy_score, question_text)
            Returns (None, 0.0, None) if no questions remain.
        """
        attr, score = choose_best_question(
            belief_state.candidates,
            self.attributes,
            belief_state.asked,
        )

        if attr is None:
            return None, 0.0, None

        question_text = self.questions.get(attr, attr)
        return attr, score, question_text
