"""
data_loader.py
--------------
Responsible for reading the CSV data files and returning structured,
ready-to-use objects. Nothing else lives here — no game logic, no entropy.

If the data format ever changes (e.g. Phase 2 adds embedding vectors),
this is the only file that needs to change.
"""

import pandas as pd
from pathlib import Path


def load_game_data(data_dir: str = "data") -> dict:
    """
    Loads word_attribute.csv and attribute_question.csv from the data directory.

    Returns a dict with:
        - df         : full DataFrame of objects and their binary attributes
        - objects    : list of object names (e.g. ["Elephant", "Chair", ...])
        - attributes : list of attribute column names (e.g. ["is_alive", "is_big", ...])
        - questions  : dict mapping attribute name -> natural language question
                       (e.g. {"is_alive": "Is it alive?", ...})
    """
    data_path = Path(data_dir)

    df = pd.read_csv(data_path / "word_attribute.csv")
    qdf = pd.read_csv(data_path / "attribute_question.csv")

    objects = df["Name"].tolist()
    attributes = df.columns[2:].tolist()  # skip Index and Name columns
    questions = dict(zip(qdf["Attributes"], qdf["Questions"]))

    return {
        "df": df,
        "objects": objects,
        "attributes": attributes,
        "questions": questions,
    }
