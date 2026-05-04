"""
game.py
-------
Responsible for the game loop: prompting the user, displaying the reasoning
trace, and outputting the final result.

No data loading, no entropy math, no filtering logic — this module
orchestrates the three agents. It knows the rules of the game, not the
internals of any system.
"""

from belief_agent import BeliefAgent, BeliefState
from question_agent import QuestionAgent
from guesser_agent import GuesserAgent, GuessResult
from conceptnet import ConceptNetClient


def display_intro(objects: list[str]) -> None:
    """Prints the opening prompt listing all objects the player can think of."""
    print("Think of one of these objects:")
    print(", ".join(objects))
    print("\nAnswer the following questions with 'y' or 'n'.")
    print("-" * 50)


def display_reasoning_trace(
    entropy_score: float,
    candidates: list[str],
    df_subset=None,
) -> None:
    """
    Prints the reasoning trace after each answer.
    Phase 2: Shows similarity scores alongside candidates when 5 or fewer remain.
    """
    count = len(candidates)
    if count <= 5 and df_subset is not None and "score" in df_subset.columns:
        scored = []
        for _, row in df_subset.iterrows():
            scored.append(f"{row['Name']} ({row['score']:.2f})")
        names = ", ".join(scored)
        print(f"  → Entropy: {entropy_score:.3f} | Candidates remaining: {count} ({names})")
    elif count <= 5:
        names = ", ".join(candidates)
        print(f"  → Entropy: {entropy_score:.3f} | Candidates remaining: {count} ({names})")
    else:
        print(f"  → Entropy: {entropy_score:.3f} | Candidates remaining: {count}")


def display_result(result: GuessResult, question_count: int) -> None:
    """Prints the final guess or fallback message."""
    print("-" * 50)

    if result.guessed:
        print(f"\nI guess you are thinking of: {result.name}")
        print(f"Got it in {question_count} question(s).")
    elif result.alternatives:
        names = ", ".join(result.alternatives)
        print(f"\nI'm not certain yet, but here are my best guesses:")
        print(names)
    else:
        print("\nI couldn't guess your object. Maybe it's not in the database or the answers didn't match.")


def run_game(data: dict) -> None:
    """
    Main game loop. Accepts the data dict returned by data_loader.load_game_data().
    Orchestrates BeliefAgent, QuestionAgent, and GuesserAgent — no logic inline.
    Phase 2.5: wires ConceptNetClient into both agents for semantic questions.
    """
    df = data["df"]
    objects = data["objects"]
    attributes = data["attributes"]
    questions = data["questions"]
    vectors = data["vectors"]

    cn_client = ConceptNetClient(cache_path="data/conceptnet_cache.json")
    new_fetches = cn_client.prefetch(objects)
    if new_fetches:
        print(f"Fetched ConceptNet data for {new_fetches} word(s).")

    display_intro(objects)

    belief_agent = BeliefAgent(vectors, cn_client)
    question_agent = QuestionAgent(attributes, questions, cn_client)
    guesser_agent = GuesserAgent()

    state = BeliefState(df.copy())

    while not guesser_agent.should_guess(state) and len(state.asked) < len(attributes):
        attr, entropy_score, question_text = question_agent.select(state)

        if attr is None:
            break

        answer = input(f"Q{state.question_count + 1}: {question_text} (y/n): ").strip().lower()

        if answer not in ["y", "n"]:
            print("  Invalid input. Please answer with 'y' or 'n'.")
            continue

        state = belief_agent.update(state, attr, answer)
        display_reasoning_trace(entropy_score, state.candidate_names, state.candidates)

        if guesser_agent.should_guess(state):
            break

    result = guesser_agent.guess(state)
    display_result(result, state.question_count)
