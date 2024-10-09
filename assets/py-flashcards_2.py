import logging
from flask import Flask, render_template, session, redirect, url_for
from markdown import markdown
import re
from pathlib import Path
import random
import os
import sqlite3
from typing import List, Dict, Tuple, Optional
import inspect

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)

app.secret_key = os.environ.get("FLASHCARDS_SECRET_KEY")

# Database file path
DB_PATH = "./flashcards.db"
k_QAFolder = "./static/md"


# ----------------------------------------------------------------------
# Parse the markdown files to extract question-answer pairs
def parse_markdown(markdown_text: str) -> List[Dict[str, str]]:
    """Parse a markdown text to extract question-answer pairs.

    Args:
        markdown_text (str): The raw markdown text.

    Returns:
        List[Dict[str, str]]: A list of dictionaries with questions and answers.
    """

    app.logger.info(f"{inspect.stack()[0][3]}()")
    markdown_text = re.sub(r"<!--.*?-->", "", markdown_text, flags=re.DOTALL)
    pattern = re.compile(r"Question\s*:\s*(.*?)\nAnswer\s*:\s*(.*?)(?=\nQuestion|\Z)", re.DOTALL)
    matches = pattern.findall(markdown_text)
    return [
        {"question": "###Question :\n" + match[0].strip(), "answer": "###Answer :\n" + match[1].strip()}
        for match in matches
    ]


# ----------------------------------------------------------------------
def load_qa_files(directory: str) -> List[Dict[str, str]]:
    """Load and parse markdown files from the specified directory.

    Args:
        directory (str): The directory containing markdown files.

    Returns:
        List[Dict[str, str]]: A list of question-answer pairs extracted from markdown files.
    """

    app.logger.info(f"{inspect.stack()[0][3]}()")
    qa_pairs = []
    qa_files = [file for file in Path(directory).iterdir() if file.is_file()]
    for qa_file in qa_files:
        try:
            with qa_file.open("r", encoding="utf-8") as f:
                markdown_text = f.read()
                qa_pairs.extend(parse_markdown(markdown_text))
        except Exception as e:
            print(f"Error reading file {qa_file.name}: {e}")
    return qa_pairs


# ----------------------------------------------------------------------
def create_db() -> None:
    """Create the SQLite database and populate it with questions and answers from markdown files."""

    app.logger.info(f"{inspect.stack()[0][3]}()")
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS flashcards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL
        )
        """
        )
        qa_pairs = load_qa_files(k_QAFolder)  # Parse markdown before storing
        for qa in qa_pairs:
            cursor.execute("INSERT INTO flashcards (question, answer) VALUES (?, ?)", (qa["question"], qa["answer"]))
        conn.commit()


# ----------------------------------------------------------------------
# SQLite database setup
def init_db() -> None:
    """Initialize the SQLite database, creating it if it doesn't exist."""

    app.logger.info(f"{inspect.stack()[0][3]}()")
    if not os.path.exists(DB_PATH):
        create_db()


# ----------------------------------------------------------------------
def get_random_flashcard(exclude_ids: List[int]) -> Optional[Tuple[int, str, str]]:
    """Get a random flashcard from the database, excluding the ones already seen.

    Args:
        exclude_ids (List[int]): List of flashcard IDs to exclude from selection.

    Returns:
        Optional[Tuple[int, str, str]]: A tuple containing the flashcard ID, question, and answer, or None if no flashcards remain.
    """

    app.logger.info(f"{inspect.stack()[0][3]}()")
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        query = "SELECT id, question, answer FROM flashcards WHERE id NOT IN ({seq}) ORDER BY RANDOM() LIMIT 1".format(
            seq=",".join(["?"] * len(exclude_ids)) if exclude_ids else "SELECT id FROM flashcards"
        )
        cursor.execute(query, exclude_ids)
        return cursor.fetchone()


# ----------------------------------------------------------------------
# Flask routes
@app.route("/")
def index() -> str:
    """Main route to display a random question and answer.

    Returns:
        str: Rendered HTML with the question and answer or a message if no questions remain.
    """

    app.logger.info(f"{inspect.stack()[0][3]}()")

    # Initialize session for unseen and seen question IDs
    if "seen_ids" not in session:
        session["seen_ids"] = []

    # Fetch a random flashcard from the database
    flashcard = get_random_flashcard(session["seen_ids"])

    if flashcard:
        current_QA = {"id": flashcard[0], "question": flashcard[1], "answer": flashcard[2]}
        session["seen_ids"].append(current_QA["id"])  # Add this question to seen list

        Q_html = markdown(current_QA["question"], extensions=["extra", "codehilite", "sane_lists"])
        A_html = markdown(current_QA["answer"], extensions=["extra", "codehilite", "sane_lists"])

        return render_template("index.html", Q_html=Q_html, A_html=A_html)
    else:
        return render_template("index.html", Q_html="No more questions.", A_html="")


# ----------------------------------------------------------------------
@app.route("/next")
def next() -> str:
    """Route to go to the next question.

    Returns:
        str: Redirect to the index route.
    """

    app.logger.info(f"{inspect.stack()[0][3]}()")
    return redirect(url_for("index"))


# ----------------------------------------------------------------------
# Application startup
if __name__ == "__main__":

    app.logger.info("main()")

    init_db()  # Initialize or create the database if it doesn't exist
    app.debug = True
    app.run()
