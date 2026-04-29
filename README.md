# neural-20-questions

> *Intelligence is not just about knowing the answer — it's about asking the right question.*

---

I originally wanted to build a program that uses a neural network to guess a word through yes/no questions — one that could learn from user interactions and draw its own conclusions over time. Before I could get there, I needed to understand the mechanics. So I started simpler: a manually-defined decision tree with a fixed list of 20 everyday objects. That constraint turned out to be the right call. It gave me something to run, observe, and question — and those questions led me somewhere more interesting than I expected.

At its core, this project is about **efficient information extraction under uncertainty**. Given a space of possible objects, the system must ask a sequence of yes/no questions that maximally reduce uncertainty until it can confidently identify the target. Traditional implementations rely on static decision trees, predefined question sequences, and hard-coded logic. They don't adapt, they don't generalize, and they have no principled way to select optimal questions. This project reframes 20 Questions as a machine learning and information theory problem.

---

## Demo

```
Think of one of these objects:
Elephant, Chair, Eagle, Person, Shark, Wrench, Coin, Dog, Hat, Jeans,
Cat, Pencil, House, Bee, Chicken, Cow, Bed, Sandwich, Water, Goldfish

Answer the following questions with 'y' or 'n'.
--------------------------------------------------
Q1: Is it alive? (y/n): y
  → Entropy: 1.000 | Candidates remaining: 10
Q2: Is it big? (y/n): y
  → Entropy: 0.881 | Candidates remaining: 3 (Elephant (0.90), Cow (0.90), Shark (0.68))
Q3: Does it live in water? (y/n): n
  → Entropy: 0.918 | Candidates remaining: 2 (Elephant (1.00), Cow (1.00))
Q4: Can you eat it? (y/n): n
  → Entropy: 1.000 | Candidates remaining: 1 (Elephant (1.00))
--------------------------------------------------
I guess you are thinking of: Elephant
Got it in 4 question(s).
```

The reasoning trace now shows two signals: the entropy score of the chosen question (how informative the split was) and a similarity score for each candidate (how semantically consistent it is with the surviving cluster). Notice Shark scoring 0.68 against Elephant and Cow at 0.90 — the system recognized the semantic outlier before the next boolean filter confirmed it.

---

## How It Works

When I first ran Phase 1, it worked — but I noticed something right away. The questions were being asked in the order they appeared in the table. There was no rhyme or reason to the sequence. If the bot asked "Is it alive?" and got a *no*, it would still ask "Can it swim?" two questions later. I wanted the bot to ask questions in order of relevancy — to eliminate as many candidates as possible with each question. I started asking: *is there a way to optimize the question pathway so it asks as few questions as possible?*

That question led me to entropy.

Each object is represented as a vector of binary attributes (e.g. `is_alive`, `can_fly`, `is_food`). At each turn, the bot selects the attribute with the highest **Shannon binary entropy** across the remaining candidates:

```
H(p) = -p·log₂(p) - (1-p)·log₂(1-p)
```

Entropy is maximized at **1.0** when exactly half the candidates have the attribute — a perfectly even split that eliminates half the possibilities no matter how the user answers. It drops to **0** when all remaining candidates share the same value, meaning the question tells you nothing new.

This transforms the system from *"ask a reasonable question"* to *"ask the most informative question mathematically."* The best question is the one that most evenly splits the hypothesis space.

This is a **greedy** strategy — it picks the locally optimal question at each step without looking ahead. That's a known limitation and a direction for future work.

### Phase 2: Semantic Scoring Layer

Phase 2 adds a semantic intelligence layer on top of the entropy-based question selector. Each candidate object is now represented as a 300-dimensional word vector loaded from spaCy's `en_core_web_md` model. After each boolean filter step, the system:

1. Builds a **centroid** from the surviving candidates' word vectors
2. Scores each survivor by **cosine similarity** to that centroid
3. Ranks and displays candidates by score in the reasoning trace

The boolean filter still drives elimination — that's what keeps the game correct. The similarity scores add a semantic ranking layer on top: they surface which candidates are most semantically consistent with the cluster that survived, and which are outliers likely to be eliminated soon.

The Shark example above illustrates this directly. After "Is it big? → yes", the surviving candidates are Elephant, Cow, and Shark. Elephant and Cow cluster tightly in semantic space (large land mammals); Shark scores noticeably lower (0.68 vs 0.90) because its vector pulls toward aquatic and predatory associations. The system flags it as the semantic outlier one question before "Does it live in water?" confirms it.

This is a form of **soft belief updating** — each answer doesn't just cut the candidate list, it reshapes the system's semantic picture of what the target probably is.

---

## Project Structure

```
neural-20-questions/
├── data/
│   ├── word_attribute.csv          # Objects and their binary attribute vectors
│   └── attribute_question.csv     # Maps attribute names to natural language questions
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # Reads CSVs, precomputes spaCy vectors, returns data dict
│   ├── entropy.py                  # Shannon entropy calculation + question selector
│   ├── candidate_filter.py        # Hybrid boolean + cosine similarity filtering
│   ├── game.py                    # Game loop, I/O, reasoning trace (GameRunner orchestrator)
│   ├── question_agent.py          # QuestionAgent — owns question selection via entropy
│   ├── belief_agent.py            # BeliefAgent — owns candidate state (BeliefState dataclass)
│   └── guesser_agent.py           # GuesserAgent — owns stopping condition (GuessResult dataclass)
├── archive/
│   ├── v2/                        # Pre-agent architecture preserved
│   │   ├── __init__.py
│   │   └── game.py
│   ├── v3/                        # Pre-Phase 2 files preserved
│   │   ├── candidate_filter.py
│   │   ├── data_loader.py
│   │   └── game.py
│   ├── notebooks/
│   │   ├── 20_Questions_P1.ipynb       # v1 — initial prototype
│   │   ├── 20_Questions_P1.5.ipynb     # v1.5 — reasoning trace added
│   │   └── 20_Questions_P1.5.1.ipynb  # v1.5.1 — final notebook version
│   └── README_v1.md               # Original README preserved verbatim
├── main.py                        # Entry point — run this to play
├── requirements.txt
└── README.md
```

---

## Setup

**Requirements:** Python 3.11+, conda (recommended) or any virtual environment manager

**Using conda:**
```bash
git clone https://github.com/DDaileg/neural-20-questions.git
cd neural-20-questions
conda create -n 20q_env python=3.11
conda activate 20q_env
pip install -r requirements.txt
python -m spacy download en_core_web_md
python main.py
```

**Using venv:**
```bash
git clone https://github.com/DDaileg/neural-20-questions.git
cd neural-20-questions
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_md
python main.py
```

The game starts immediately in your terminal.

---

## Project Journey

This project didn't start as a clean architecture. It started as a question: *can I build something that learns?* What followed was a series of working prototypes, each one exposing something I didn't know yet — and sending me back to learn it before I could move forward.

### v1 — Initial Prototype (`20_Questions_P1.ipynb`)
The first working version. A single Jupyter notebook with a fixed list of 20 objects, 15 binary attributes, and a game loop that asked questions in table order. No intelligence in the sequencing — just mechanics. The goal was to prove the concept worked before adding any complexity.

It worked. And the moment it worked, I noticed the problem.

### v1.5 — Information-Theoretic Optimization (`20_Questions_P1.5.ipynb`)
After running v1, I wrote down what bothered me: the questions had no order of relevancy. I put that observation into an LLM and got a recommendation — entropy-based question selection using decision tree learning. Before implementing it, I set myself a learning agenda: understand Shannon entropy, understand how algorithms implement it, understand decision tree learning. Then I built it.

This version also added a reasoning trace that made the decision process visible after each answer. At this point I briefly explored pivoting the game to guess someone's *career* instead of an object — tested the idea, decided the mechanics weren't different enough to justify the detour, and reverted. Staying with the original scope was the right call.

### v1.5.1 — Final Notebook Version (`20_Questions_P1.5.1.ipynb`)
Refinements to the game loop, input validation, and output formatting. The last notebook-only version before the architectural overhaul.

### v2 — Clean Architecture (`main.py` + `src/`)
A full restructure from notebook to modular Python package, informed by a multi-model design review (Claude + ChatGPT). The notebook had all logic — data loading, entropy math, filtering, game loop — living in the same place. v2 separates those concerns across four modules so each one can evolve independently.

Key decisions:
- **`data_loader.py`** — data access isolated; single return dict with named keys
- **`entropy.py`** — entropy math fully decoupled from game state; `attributes` passed as a parameter instead of referenced as a global
- **`candidate_filter.py`** — filtering extracted and independently testable; `reset_index` applied for clean downstream DataFrames
- **`game.py`** — orchestrates the other modules; display functions separated for future Streamlit compatibility
- **`main.py`** — single entry point; `python main.py` runs the game from the project root

Behavior is identical to v1.5.1. Structure is production-ready.

### v3 — Multi-Agent Architecture
The v2 modules were clean, but `game.py` was still making decisions it shouldn't own — calling entropy directly, holding game state as loose local variables, embedding the stopping condition in a `while` loop. v3 introduces three explicit agents, each with a single defined responsibility.

`QuestionAgent` owns question selection. `BeliefAgent` owns candidate state — it introduces `BeliefState`, an immutable data object that replaces the three separate variables (`filtered_df`, `asked`, `question_count`) the loop was managing manually. `GuesserAgent` owns the stopping condition and produces a typed `GuessResult`, separating the *decision* from the *display*. `GameRunner` in `game.py` is now a pure orchestrator: it reads input, prints output, and passes data between agents. It makes no decisions itself.

Each future phase has a designated seam:
- Phase 2 (spaCy embeddings) → `BeliefAgent.update()` and `candidate_filter.py`
- Phase 3 (neural question selector) → `GuesserAgent` + swap `entropy.py`
- Streamlit interface → subclass `GameRunner`, override display and input

Behavior is identical to v1.5.1. Architecture is extensible by design.

### v4 — Phase 2: Semantic Intelligence Layer
Phase 2 upgrades `candidate_filter.py` and `data_loader.py` to add a semantic scoring layer on top of the boolean filtering system.

`data_loader.py` now loads spaCy's `en_core_web_md` model at startup and precomputes a 300-dimensional word vector for each of the 20 candidate objects. These vectors are stored in the data dict and passed through the system without any other module needing to import spaCy directly.

`candidate_filter.py` now runs a two-step process on every turn: hard boolean filter first (same as before), then centroid-based cosine similarity scoring on the survivors. The centroid is computed from the surviving candidates' vectors after each filter step. Each survivor is scored by how close it sits to that centroid in semantic space, and the scores are attached to the DataFrame for display in the reasoning trace.

The key design decision: similarity scoring ranks survivors, it doesn't eliminate them. Boolean filtering remains the gate. This keeps the game correct while making the reasoning trace semantically meaningful — you can watch the system's confidence distribute across candidates as the search narrows.

`game.py` required only one change: passing `vectors` into `filter_candidates()`. No other module was touched.

---

## What I Learned

I started this project with two questions I had written down:

> *What are neural networks? What is taxonomy, and what role does it play in linguistics?*

I haven't fully answered either of them yet — this project is still in progress. But working through it taught me things I didn't know to ask about when I started.

**Shannon entropy as a measure of uncertainty.** I looked it up because a program I built was asking questions in the wrong order. Once I understood it, I realized it was the right mathematical tool for exactly the problem I had — and that it explained *why* certain questions are better than others, not just empirically but provably.

**Binary feature engineering.** Designing the attribute table for 20 objects was more thought-intensive than I expected. Choosing which features to include, how to phrase them, and how to make them discriminating without being redundant is a form of manual taxonomy design. That work directly shapes how well the entropy selector performs. Good taxonomies reduce the number of questions needed — that's a linguistic design insight that took building the system to fully understand.

**Greedy vs. optimal search.** The current selector is greedy — locally optimal at each step, but blind to how a question sequence plays out across multiple turns. A lookahead or beam search approach could improve performance by considering sequences rather than individual questions. That gap is deliberate and documented.

**Clean architecture as a scaling decision.** The notebook version worked. Refactoring it wasn't about fixing something broken — it was about making sure the entropy module could be swapped for an embedding-based selector in Phase 2 without touching the game loop. That kind of forward-looking structure is something I've started thinking about before I write the first line of any new module.

**Word vectors as semantic geometry.** In Phase 2, I ran into a design failure before getting to the working version. My first attempt used similarity scoring as the sole elimination mechanism — no boolean filter. Everything stayed above the threshold because cosine similarity in a dense vector space doesn't naturally produce clean hard separations. The fix was to keep boolean filtering as the gate and use similarity for ranking only. That distinction — between a hard decision boundary and a soft scoring signal — is something I now think about explicitly when designing any filtering step.

---

## What's Next

### Phase 2.5 — Hybrid Reasoning with ConceptNet

Rather than manually expanding the attribute list, Phase 2.5 will use **ConceptNet** to automatically extract features from real-world relational knowledge. ConceptNet's relation types — `IsA`, `UsedFor`, `HasProperty`, `CapableOf`, `AtLocation` — map naturally to yes/no question templates:

| Relation | Example | Question |
|---|---|---|
| `IsA` | dog → animal | "Is it a type of animal?" |
| `HasProperty` | ice → cold | "Is it cold?" |
| `UsedFor` | hammer → hitting nails | "Is it used for hitting?" |
| `CapableOf` | bird → flying | "Can it fly?" |

This means the system can generate its own questions for objects it hasn't seen before — and validate them against spaCy's semantic space to ensure they're discriminative.

### Full Roadmap

- [x] **Phase 1 — Manual decision tree:** fixed object list, binary attributes, game loop
- [x] **Phase 1.5 — Entropy-based question selection:** Shannon entropy replaces table-order questioning; reasoning trace added
- [x] **Phase 2 — Semantic scoring layer:** spaCy word vectors + cosine similarity scoring on survivors; semantic ranking visible in reasoning trace
- [ ] **Phase 2.5 — ConceptNet integration:** automated feature extraction from relational knowledge; dynamically generated yes/no questions from relation templates
- [ ] **Phase 3 — Neural question selector:** train an MLP to predict the best next question from the current answer state, replacing the greedy entropy selector
- [ ] **Phase 4 — Online learning:** update the knowledge base when the bot guesses wrong; new objects and attributes added through gameplay
- [ ] **Streamlit interface:** browser-based version for shareable demos

---

## Long-Term Vision

The mechanics of this project aren't specific to a guessing game. An adaptive questioning engine — one that selects the most informative question given what it already knows — is a general tool. The same architecture could:

- **Diagnose student misconceptions** by asking targeted questions to locate where understanding breaks down (directly related to a teaching assistant project currently in development)
- **Assist in decision triage** by narrowing a large possibility space through structured yes/no inquiry
- **Power conversational agents** with goal-directed reasoning rather than passive response generation

The game is the proof of concept. The underlying system is the point.
