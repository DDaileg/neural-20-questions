"""
main.py
-------
Entry point for Neural 20 Questions.

Run with:
    python main.py

That's it. All logic lives in src/.
"""

import sys
from pathlib import Path

# Allow imports from src/ without installing the package
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader import load_game_data
from game import run_game


def main():
    data = load_game_data(data_dir="data")
    run_game(data)


if __name__ == "__main__":
    main()
