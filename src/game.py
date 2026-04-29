"""
game.py
-------
Responsible for the game loop: prompting the user, displaying the reasoning
trace, and outputting the final result.

No data loading, no entropy math, no filtering logic — this module
orchestrates the other three. It knows the rules of the game, not the
internals of any system.

Phase 2 change: vectors dict passed from data into filter_candidates().
"""

from entropy import choose_best_question
from candidate_filter import filter_candidates, get_candidate_names


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


def display_result(filtered_df, question_count: int) -> None:
    """Prints the final guess or fallback message."""
    print("-" * 50)

    if len(filtered_df) == 1:
        guess = filtered_df["Name"].values[0]
        print(f"\nI guess you are thinking of: {guess}")
        print(f"Got it in {question_count} question(s).")
    elif len(filtered_df) > 1:
        names = ", ".join(get_candidate_names(filtered_df))
        print(f"\nI'm not certain yet, but here are my best guesses:")
        print(names)
    else:
        print("\nI couldn't guess your object. Maybe it's not in the database or the answers didn't match.")


def run_game(data: dict) -> None:
    """
    Main game loop. Accepts the data dict returned by data_loader.load_game_data().

    Phase 2: passes vectors dict into filter_candidates() for similarity scoring.
    """
    df = data["df"]
    objects = data["objects"]
    attributes = data["attributes"]
    questions = data["questions"]
    vectors = data["vectors"]  # Phase 2

    display_intro(objects)

    filtered_df = df.copy()
    asked = set()
    question_count = 0

    while len(filtered_df) > 1 and len(asked) < len(attributes):
        attr, entropy_score = choose_best_question(filtered_df, attributes, asked)

        if attr is None:
            break

        question_count += 1
        question_text = questions.get(attr, attr)

        answer = input(f"Q{question_count}: {question_text} (y/n): ").strip().lower()

        if answer not in ["y", "n"]:
            print("  Invalid input. Please answer with 'y' or 'n'.")
            question_count -= 1
            continue

        filtered_df = filter_candidates(filtered_df, attr, answer, vectors)
        asked.add(attr)

        candidates = get_candidate_names(filtered_df)
        display_reasoning_trace(entropy_score, candidates, filtered_df)

        if len(filtered_df) == 1:
            break

    display_result(filtered_df, question_count)
