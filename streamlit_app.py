"""Browser demo for Neural 20 Questions.

Run from the project root with:
    streamlit run streamlit_app.py
"""

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from belief_agent import BeliefAgent, BeliefState
from data_loader import load_game_data
from guesser_agent import GuesserAgent
from question_agent import QuestionAgent


@st.cache_resource(show_spinner="Loading the language model…")
def load_data():
    return load_game_data(data_dir=str(ROOT / "data"))


def reset_game(data):
    st.session_state.state = BeliefState(data["df"].copy())
    st.session_state.trace = []


st.set_page_config(page_title="Neural 20 Questions", page_icon="❓", layout="centered")
st.title("Neural 20 Questions")
st.caption("An explainable guessing game using entropy, hard filtering, and semantic ranking.")

data = load_data()
belief_agent = BeliefAgent(data["vectors"])
question_agent = QuestionAgent(data["attributes"], data["questions"])
guesser_agent = GuesserAgent()

if "state" not in st.session_state:
    reset_game(data)

state = st.session_state.state

with st.expander("Candidate objects", expanded=False):
    st.write(", ".join(data["objects"]))

if guesser_agent.should_guess(state):
    result = guesser_agent.guess(state)
    if result.guessed:
        st.success(f"My guess: **{result.name}**")
        st.write(f"Solved in {state.question_count} question(s).")
    else:
        st.error("No candidates remain. One or more answers conflict with the data.")
    if st.button("Play again", type="primary", use_container_width=True):
        reset_game(data)
        st.rerun()
else:
    attr, entropy_score, question_text = question_agent.select(state)
    st.metric("Candidates remaining", state.candidate_count)
    st.subheader(f"Question {state.question_count + 1}")
    st.write(f"### {question_text}")
    yes, no = st.columns(2)
    answer = None
    if yes.button("Yes", type="primary", use_container_width=True):
        answer = "y"
    if no.button("No", use_container_width=True):
        answer = "n"

    if answer:
        new_state = belief_agent.update(state, attr, answer)
        survivors = new_state.candidates
        trace_row = {
            "question": question_text,
            "answer": "Yes" if answer == "y" else "No",
            "entropy": round(entropy_score, 3),
            "remaining": new_state.candidate_count,
            "candidates": ", ".join(new_state.candidate_names[:8]),
        }
        st.session_state.trace.append(trace_row)
        st.session_state.state = new_state
        st.rerun()

if st.session_state.trace:
    st.divider()
    st.subheader("Reasoning trace")
    st.dataframe(st.session_state.trace, hide_index=True, use_container_width=True)

st.divider()
st.caption("Boolean attributes determine elimination. spaCy similarity scores rank survivors; they do not override the hard filter.")
