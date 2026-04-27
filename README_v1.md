# neural-20-questions
An exploration of information theory through an interactive game.
This game contains AI that plays 20 Questions using **entropy-based question selection** to narrow down a word in as few questions as possible.
 
Instead of asking questions in a fixed order, the bot calculates which question will split the remaining candidates most evenly — maximizing information gained with every answer. Each question is chosen because it is the most useful one to ask *right now*, given what the bot already knows.
 
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
 
Each object in the dataset is represented as a vector of binary attributes (e.g. `is_alive`, `can_fly`, `is_food`). At each turn, the bot selects the attribute with the highest **Shannon binary entropy** across the remaining candidates:
 
```
H(p) = -p·log₂(p) - (1-p)·log₂(1-p)
```
 
Entropy is maximized at **1.0** when exactly half the candidates have the attribute (a perfectly even split), and drops to **0** when all remaining candidates share the same value (no new information). By always picking the highest-entropy question, the bot minimizes the expected number of questions needed to identify the object.
 
This is a **greedy** strategy — it picks the locally optimal question at each step. It does not look ahead across multiple questions.
 
---
 
## Project Structure
 
```
neural-20-questions/
├── data/
│   ├── word_attribute.csv       # Objects and their binary attribute vectors
│   └── attribute_question.csv  # Maps attribute names to natural language questions
├── 20_Questions_P1.5.ipynb      # Main notebook with entropy selection + reasoning trace
└── README.md
```
 
---
 
## Setup
 
**Requirements:** Python 3.11+, Jupyter Notebook
 
```bash
git clone https://github.com/DDaileg/neural-20-questions.git
cd neural-20-questions
pip install pandas numpy jupyter
jupyter notebook
```
 
Open `20_Questions_P1.5.1.ipynb` and run all cells. The game loop is in the final cell.
 
---
 
## What I Learned
 
This project started as a simple decision tree and evolved into an applied study of **information theory**. Key concepts explored:
 
- **Shannon entropy** as a measure of uncertainty — and why maximizing it leads to efficient question selection
- **Binary feature engineering** — representing real-world concepts as structured attribute vectors (a form of manual taxonomy design)
- **Greedy vs. optimal search** — the current selector is greedy (locally optimal). A lookahead or beam search approach could improve performance by considering question sequences rather than individual questions
- **Visible reasoning** — designing AI systems that show their work, not just their answers
---
 
## Roadmap
 
- [ ] **Phase 2 — Semantic embeddings:** Replace hand-coded binary attributes with spaCy word vectors. Objects represented as 300-dimensional semantic vectors; similarity structure emerges from the language model rather than manual feature design.
- [ ] **Phase 3 — Neural question selector:** Train an MLP to predict the best next question from the current answer state, replacing the greedy entropy selector.
- [ ] **Phase 4 — Online learning:** Update the knowledge base when the bot guesses wrong. New objects and attributes added through gameplay.
- [ ] **Streamlit interface:** Browser-based version for shareable demos.
