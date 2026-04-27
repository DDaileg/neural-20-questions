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
Q1: Is it alive? (y/n): n
  → Entropy: 1.000 | Candidates remaining: 10
Q2: Is it big? (y/n): n
  → Entropy: 0.722 | Candidates remaining: 8
Q3: Can you eat it? (y/n): y
  → Entropy: 0.811 | Candidates remaining: 2 (Sandwich, Water)
Q4: Is it liquid? (y/n): n
  → Entropy: 1.000 | Candidates remaining: 1 (Sandwich)
--------------------------------------------------
I guess you are thinking of: Sandwich
Got it in 4 question(s).
```

The reasoning trace after each answer shows the entropy score of the chosen question and how many candidates remain — making the decision process visible at every step.

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

---

## Project Structure

```
neural-20-questions/
├── data/
│   ├── word_attribute.csv          # Objects and their binary attribute vectors
│   └── attribute_question.csv     # Maps attribute names to natural language questions
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # Reads CSVs, returns structured data dict
│   ├── entropy.py                  # Shannon entropy calculation + question selector
│   ├── candidate_filter.py        # Filters candidate set based on yes/no answers
│   └── game.py                    # Game loop, I/O, reasoning trace
├── archive/
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

**Requirements:** Python 3.11+

```bash
git clone https://github.com/DDaileg/neural-20-questions.git
cd neural-20-questions
pip install -r requirements.txt
python main.py
```

That's it. The game starts immediately in your terminal.

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
- **`candidate_filter.py`** — filtering extracted and independently testable; `reset_index` applied for clean downstream DataFrames; Phase 2 swaps the boolean filter for soft similarity filtering here without touching anything else
- **`game.py`** — orchestrates the other modules; display functions separated for future Streamlit compatibility
- **`main.py`** — single entry point; `python main.py` runs the game from the project root

Behavior is identical to v1.5.1. Structure is production-ready.

---

## What I Learned

I started this project with two questions I had written down:

> *What are neural networks? What is taxonomy, and what role does it play in linguistics?*

I haven't fully answered either of them yet — this project is still in progress. But working through it taught me things I didn't know to ask about when I started.

**Shannon entropy as a measure of uncertainty.** I looked it up because a program I built was asking questions in the wrong order. Once I understood it, I realized it was the right mathematical tool for exactly the problem I had — and that it explained *why* certain questions are better than others, not just empirically but provably.

**Binary feature engineering.** Designing the attribute table for 20 objects was more thought-intensive than I expected. Choosing which features to include, how to phrase them, and how to make them discriminating without being redundant is a form of manual taxonomy design. That work directly shapes how well the entropy selector performs. Good taxonomies reduce the number of questions needed — that's a linguistic design insight that took building the system to fully understand.

**Greedy vs. optimal search.** The current selector is greedy — locally optimal at each step, but blind to how a question sequence plays out across multiple turns. A lookahead or beam search approach could improve performance by considering sequences rather than individual questions. That gap is deliberate and documented.

**Clean architecture as a scaling decision.** The notebook version worked. Refactoring it wasn't about fixing something broken — it was about making sure the entropy module could be swapped for an embedding-based selector in Phase 2 without touching the game loop. That kind of forward-looking structure is something I've started thinking about before I write the first line of any new module.

---

## What's Next

### Phase 2 — Semantic Intelligence Layer

Phase 2 has been on my mind for a while. The binary attribute system works, but it has a hard ceiling: every feature is hand-coded, and the system can only guess objects that already exist in the dataset. Phase 2 replaces that with something more flexible.

The plan is to integrate **spaCy** (specifically `en_core_web_md`) to represent each object as a 300-dimensional semantic vector. Instead of filtering by boolean match, the system will use **cosine similarity** to measure how close the remaining candidates are to the inferred "truth vector" as each answer narrows the search space. Each yes/no answer isn't just a filter — it becomes a transformation of belief in semantic space.

I also want to build **data visualizations** from this: a cosine similarity heatmap across all objects, and a 2D projection (via t-SNE or PCA) showing how objects cluster semantically. I've worked with correlation matrices before, and I expect there's a meaningful analog here — some kind of visual map of how the objects relate to each other in semantic space.

The architecture is already set up for this swap. `entropy.py` and `candidate_filter.py` are isolated enough that Phase 2 changes those two modules and nothing else.

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

- [ ] **Phase 2 — Semantic embeddings:** spaCy vectors replace binary attributes; cosine similarity replaces boolean filtering; similarity heatmap + t-SNE visualization
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
