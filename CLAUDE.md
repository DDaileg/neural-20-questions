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
- `QuestionAgent` — selects the best question across two pools: static boolean attributes (entropy from DataFrame columns) and ConceptNet-generated semantic questions (entropy from relation coverage). CN wins on ties. When a CN question wins, it is resolved to its mapped static attribute via `_CN_TO_ATTR` in `question_agent.py` before being returned — so `select()` always returns a static attribute column name as the filter key, with the CN question text preserved for display.
- `BeliefAgent` + `BeliefState` — immutable state tracking. `update()` always calls `filter_candidates()` with a static attribute column name. It never calls `filter_candidates_by_conceptnet()`.
- `GuesserAgent` — stopping condition + `GuessResult` dataclass.

**`candidate_filter.py`**: Two filter functions. `filter_candidates()` uses a boolean column as the gate; this is the only one called in the active game loop. `filter_candidates_by_conceptnet()` is retained in the codebase for future use but is not called during gameplay. Both apply centroid + cosine scoring as a display-only layer on top.

**`conceptnet.py`**: `ConceptNetClient` fetches edges from `api.conceptnet.io`, caches to `data/conceptnet_cache.json` (never re-fetches a cached word). Exposes `get_question_candidates(candidates)`, `get_relation_coverage(key, candidates)`, and `has_relation(word, key)`. Returns empty results gracefully on network failure. Question key format: `"cn:IsA:bird"`, `"cn:CapableOf:fly"`, etc.

**`entropy.py`**: Greedy Shannon entropy — picks the attribute that splits the remaining candidate set most evenly. Also exports `binary_entropy()` for use in the CN question scoring path.

## Data schema

`data/word_attribute.csv` — columns: `Index`, `Name`, then 15 boolean attribute columns (e.g. `is_alive`, `is_big`).  
`data/attribute_question.csv` — columns: `Attributes`, `Questions` — maps attribute name to natural language yes/no question.

## Design constraints (do not violate)

1. Boolean filter is the hard gate for all questions — cosine similarity ranks survivors, never eliminates them.
2. ConceptNet is used for question selection only, not filtering. CN questions are mapped to static attributes via `_CN_TO_ATTR` in `question_agent.py` before the filter step. `filter_candidates_by_conceptnet()` is not called in the active game loop.
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
