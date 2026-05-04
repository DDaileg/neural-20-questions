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

CN questions are used for selection only, not filtering. When a CN question
wins, it is mapped to the closest static attribute column via _CN_TO_ATTR.
The mapped static attr is returned as the filter key; the CN question text
is preserved as the display text shown to the user. CN keys with no mapping
are skipped — the static question wins instead.

Interface:
    agent = QuestionAgent(attributes, questions, cn_client)
    attr, score, text = agent.select(belief_state)
    # attr is always a static attribute column name (used for filtering)
    # text is the CN question text when CN won (what the user sees)
"""

from entropy import choose_best_question, binary_entropy

# Maps CN question keys to static attribute column names.
# Only CN questions with a known mapping compete against static questions.
# The mapped attr drives filtering; the CN question text is shown to the user.
_CN_TO_ATTR: dict[str, str] = {
    # IsA — biological categories
    "cn:IsA:animal":            "is_animal",
    "cn:IsA:mammal":            "is_animal",
    "cn:IsA:bird":              "is_animal",
    "cn:IsA:fish":              "is_animal",
    "cn:IsA:insect":            "is_animal",
    "cn:IsA:arthropod":         "is_animal",
    "cn:IsA:bug":               "is_animal",
    "cn:IsA:predator":          "is_animal",
    "cn:IsA:raptor":            "is_animal",
    "cn:IsA:poultry":           "is_animal",
    "cn:IsA:farm_animal":       "is_animal",
    "cn:IsA:marine_animal":     "is_animal",
    "cn:IsA:canine":            "is_animal",
    "cn:IsA:feline":            "is_animal",
    # IsA — special animal categories
    "cn:IsA:pet":               "is_pet",
    "cn:IsA:human":             "is_human",
    # IsA — objects
    "cn:IsA:furniture":         "is_furniture",
    "cn:IsA:seat":              "is_furniture",
    "cn:IsA:tool":              "is_tool",
    "cn:IsA:hand_tool":         "is_tool",
    "cn:IsA:writing_instrument": "can_write",
    "cn:IsA:stationery":        "can_write",
    "cn:IsA:clothing":          "is_clothes",
    "cn:IsA:headwear":          "is_clothes",
    "cn:IsA:pants":             "is_clothes",
    "cn:IsA:trousers":          "is_clothes",
    "cn:IsA:accessory":         "is_accessory",
    "cn:IsA:food":              "is_food",
    "cn:IsA:snack":             "is_food",
    "cn:IsA:meal":              "is_food",
    "cn:IsA:liquid":            "is_liquid",
    "cn:IsA:drink":             "is_liquid",
    # CapableOf
    "cn:CapableOf:fly":         "can_fly",
    "cn:CapableOf:soar":        "can_fly",
    "cn:CapableOf:swim":        "can_swim",
    "cn:CapableOf:bark":        "can_bark",
    # UsedFor
    "cn:UsedFor:writing":       "can_write",
    "cn:UsedFor:drawing":       "can_write",
}


def _map_cn_to_attr(cn_key: str) -> str | None:
    """Returns the static attribute column for a CN key, or None if unmapped."""
    return _CN_TO_ATTR.get(cn_key)


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
            attr is always a static attribute column name.
            question_text is the CN question text when CN won.
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
        Finds the highest-entropy ConceptNet question for the current candidates
        that maps to an unasked static attribute.

        Returns (mapped_static_attr, score, cn_question_text), or (None, -1, None)
        if no mappable unasked CN question exists.
        """
        candidates = belief_state.candidate_names
        asked = belief_state.asked

        best_attr: str | None = None
        best_score: float = -1.0
        best_text: str | None = None

        for q_key, q_text in self.cn.get_question_candidates(candidates):
            mapped_attr = _map_cn_to_attr(q_key)
            if mapped_attr is None or mapped_attr in asked:
                continue
            coverage = self.cn.get_relation_coverage(q_key, candidates)
            score = binary_entropy(coverage)
            if score > best_score:
                best_score = score
                best_attr = mapped_attr
                best_text = q_text

        return best_attr, best_score, best_text
