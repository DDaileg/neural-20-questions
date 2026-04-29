"""
data_loader.py
--------------
Responsible for reading the CSV data files and returning structured,
ready-to-use objects. Nothing else lives here — no game logic, no entropy.

Phase 2: Now also precomputes spaCy word vectors for all candidate objects.
Vectors are loaded once here and passed through the data dict so nothing
else in the system needs to import spaCy directly.
"""

import pandas as pd
import numpy as np
import spacy
from pathlib import Path


def load_game_data(data_dir: str = "data") -> dict:
    """
    Loads word_attribute.csv and attribute_question.csv from the data directory.
    Phase 2: Also loads spaCy en_core_web_md and precomputes a 300-dim vector
    for each candidate object name.

    Returns a dict with:
        - df         : full DataFrame of objects and their binary attributes
        - objects    : list of object names (e.g. ["Elephant", "Chair", ...])
        - attributes : list of attribute column names (e.g. ["is_alive", "is_big", ...])
        - questions  : dict mapping attribute name -> natural language question
                       (e.g. {"is_alive": "Is it alive?", ...})
        - vectors    : dict mapping object name -> 300-dim numpy vector
                       (e.g. {"Elephant": array([...]), ...})
    """
    data_path = Path(data_dir)

    df = pd.read_csv(data_path / "word_attribute.csv")
    qdf = pd.read_csv(data_path / "attribute_question.csv")

    objects = df["Name"].tolist()
    attributes = df.columns[2:].tolist()  # skip Index and Name columns
    questions = dict(zip(qdf["Attributes"], qdf["Questions"]))

    # Phase 2: precompute spaCy vectors for all candidate words
    nlp = spacy.load("en_core_web_md")
    vectors = {name: nlp(name.lower()).vector for name in objects}

    return {
        "df": df,
        "objects": objects,
        "attributes": attributes,
        "questions": questions,
        "vectors": vectors,
    }
