"""
question_agent.py
-----------------
Agent responsible for selecting the next question to ask.

Phase 2.5: when a ConceptNetClient is provided, select() also builds a
pool of CN-derived questions for the current candidates, scores each by
Shannon entropy on their ConceptNet relation coverage, and picks the
overall best question across both pools.

CN questions win on tie — semantically richer questions are preferred
when they are equally informative to a boolean attribute question.

Interface:
    agent = QuestionAgent(attributes, questions, cn_client)
    attr, score, text = agent.select(belief_state)
"""

from entropy import choose_best_question, binary_entropy


class QuestionAgent:
    """
    Selects the most informative question given the current belief state.

    Args:
        attributes : full list of static attribute column names
        questions  : dict mapping attribute name -> natural language question
        cn_client  : optional ConceptNetClient; if None, only static questions
                     are used
    """

    def __init__(
        self,
        attributes: list[str],
        questions: dict[str, str],
        cn_client=None,
    ):
        self.attributes = attributes
        self.questions = questions
        self.cn = cn_client

    def select(self, belief_state: "BeliefState") -> tuple[str | None, float, str | None]:
        """
        Picks the best unasked question across static attributes and (if a
        ConceptNetClient is present) ConceptNet-generated questions.

        CN questions win on tie so semantic questions surface when equally
        informative to a boolean attribute question.

        Returns:
            (attr, entropy_score, question_text)
            Returns (None, 0.0, None) if no questions remain.
        """
        static_attr, static_score = choose_best_question(
            belief_state.candidates,
            self.attributes,
            belief_state.asked,
        )

        best_attr = static_attr
        best_score = static_score if static_attr is not None else -1.0
        best_text = self.questions.get(static_attr, static_attr) if static_attr else None

        if self.cn is not None:
            cn_attr, cn_score, cn_text = self._best_cn_question(belief_state)
            if cn_attr is not None and cn_score >= best_score:
                best_attr, best_score, best_text = cn_attr, cn_score, cn_text

        if best_attr is None:
            return None, 0.0, None

        return best_attr, best_score, best_text

    def _best_cn_question(
        self,
        belief_state: "BeliefState",
    ) -> tuple[str | None, float, str | None]:
        """
        Finds the highest-entropy ConceptNet question for the current candidates.
        Coverage = fraction of candidates that have the relation.
        """
        candidates = belief_state.candidate_names
        asked = belief_state.asked

        best_attr: str | None = None
        best_score: float = -1.0
        best_text: str | None = None

        for q_key, q_text in self.cn.get_question_candidates(candidates):
            if q_key in asked:
                continue
            coverage = self.cn.get_relation_coverage(q_key, candidates)
            score = binary_entropy(coverage)
            if score > best_score:
                best_score = score
                best_attr = q_key
                best_text = q_text

        return best_attr, best_score, best_text
