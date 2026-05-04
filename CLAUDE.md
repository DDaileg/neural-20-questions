# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Activate environment first (always required)
conda activate 20q_env

# Run the game
python main.py

# Run tests (if any exist)
python -m pytest

# Install/update deps inside env only
pip install -r requirements.txt
python -m spacy download en_core_web_md
```

Do NOT use pip outside the conda env (`20q_env`).

## Architecture

`main.py` calls `data_loader.load_game_data()` then `game.run_game(data)`. Everything else is delegated.

**Data flow**: `data_loader.py` reads two CSVs and precomputes 300-dim spaCy vectors for every candidate word. The returned `data` dict (keys: `df`, `objects`, `attributes`, `questions`, `vectors`) is the single source of truth passed through the system.

**`game.py`**: Pure orchestrator. Creates `ConceptNetClient`, `BeliefAgent`, `QuestionAgent`, and `GuesserAgent`, then runs the loop by delegating to each.

**Agent layer**:
- `QuestionAgent` — selects the best question across two pools: static boolean attributes (entropy from DataFrame columns) and ConceptNet-generated semantic questions (entropy from relation coverage). CN wins on ties.
- `BeliefAgent` + `BeliefState` — immutable state tracking. `update()` routes `"cn:..."` keys to `filter_candidates_by_conceptnet()` and all other keys to the boolean `filter_candidates()`.
- `GuesserAgent` — stopping condition + `GuessResult` dataclass.

**`candidate_filter.py`**: Two filter functions, same two-step structure. `filter_candidates()` uses a boolean column as the gate. `filter_candidates_by_conceptnet()` uses ConceptNet relation membership as the gate. Both apply centroid + cosine scoring as a display-only layer on top.

**`conceptnet.py`**: `ConceptNetClient` fetches edges from `api.conceptnet.io`, caches to `data/conceptnet_cache.json` (never re-fetches a cached word). Exposes `get_question_candidates(candidates)`, `get_relation_coverage(key, candidates)`, and `has_relation(word, key)`. Returns empty results gracefully on network failure. Question key format: `"cn:IsA:bird"`, `"cn:CapableOf:fly"`, etc.

**`entropy.py`**: Greedy Shannon entropy — picks the attribute that splits the remaining candidate set most evenly. Also exports `binary_entropy()` for use in the CN question scoring path.

## Data schema

`data/word_attribute.csv` — columns: `Index`, `Name`, then 15 boolean attribute columns (e.g. `is_alive`, `is_big`).  
`data/attribute_question.csv` — columns: `Attributes`, `Questions` — maps attribute name to natural language yes/no question.

## Design constraints (do not violate)

1. Boolean filter is the hard gate for static attribute questions — cosine similarity ranks survivors, never eliminates them.
2. ConceptNet relation membership is the gate for CN questions — same cosine scoring layer on top.
3. Question selection is greedy entropy — no lookahead or beam search. CN wins on entropy ties.
4. Invalid input does not cost a turn.
5. `GameRunner` is a pure orchestrator — no filtering, scoring, or entropy math inside `game.py`.
6. Local-first — no new cloud dependencies beyond the existing ConceptNet HTTP call (already in Phase 2.5).
7. Do not touch `archive/` — it is intentional historical record.

## Phase history

- **Phase 1**: boolean filtering only (archived in `archive/v2/`)
- **Phase 1.1**: entropy-based question selection
- **Phase 2**: hybrid filter — boolean gate + cosine similarity scoring via spaCy `en_core_web_md`
- **Phase 2.5**: ConceptNet relation integration — semantic questions alongside static attributes
- **Phase 3** (next): `GuesserAgent` confidence scoring; probabilistic early stopping before all candidates are eliminated

## Git workflow

- Branch: `main`, remote: `https://github.com/DDaileg/neural-20-questions`
- Commit messages: verb-first, lowercase, specific (`add cosine scoring to candidate_filter`, not `update code`)
