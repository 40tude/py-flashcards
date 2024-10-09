# Locally :
#   you must define the session key before to run/debug the app
#       ./secrets.ps1
#       ls env:FLASH*                 # to double check
#   python py-flashcards.py or F5
#   flask --app py-flashcards run

import os
import re
import inspect
import logging
import sqlite3
from pathlib import Path
from markdown import markdown
from typing import List, Dict, Tuple, Optional
from flask import Flask, render_template, session, redirect, url_for


# logging.basicConfig(level=logging.INFO)
# app = Flask(__name__)

# If you run the app locally you must run ./secrets.ps1 first (see above)
# In production on Heroku FLASHCARDS_SECRET_KEY must have been set manually (see readme.md)
# Without session key, Flask does not allow the app to set or access the session dictionary
# app.secret_key = os.environ.get("FLASHCARDS_SECRET_KEY")

# Database file path
k_DB_Path = "./flashcards.db"
k_QAFolder = "./static/md"


# ----------------------------------------------------------------------
# Parse the markdown files and convert to HTML
def parse_markdown_to_html(markdown_text: str) -> List[Dict[str, str]]:
    """Parse a markdown text, convert the question-answer pairs to HTML.

    Args:
        markdown_text (str): The raw markdown text.

    Returns:
        List[Dict[str, str]]: A list of dictionaries with questions and answers in HTML format.
    """

    # app.logger.info(f"{inspect.stack()[0][3]}()")
    markdown_text = re.sub(r"<!--.*?-->", "", markdown_text, flags=re.DOTALL)
    pattern = re.compile(r"Question\s*:\s*(.*?)\nAnswer\s*:\s*(.*?)(?=\nQuestion|\Z)", re.DOTALL)
    matches = pattern.findall(markdown_text)
    return [
        {
            "question_html": markdown(
                "###Question :\n" + match[0].strip(), extensions=["extra", "codehilite", "sane_lists"]
            ),
            "answer_html": markdown(
                "###Answer :\n" + match[1].strip(), extensions=["extra", "codehilite", "sane_lists"]
            ),
        }
        for match in matches
    ]


# ----------------------------------------------------------------------
def load_qa_files(directory: str) -> List[Dict[str, str]]:
    """Load and parse markdown files from the specified directory, converting them to HTML.

    Args:
        directory (str): The directory containing markdown files.

    Returns:
        List[Dict[str, str]]: A list of question-answer pairs in HTML format extracted from markdown files.
    """

    # app.logger.info(f"{inspect.stack()[0][3]}()")
    qa_pairs = []
    qa_files = [file for file in Path(directory).iterdir() if file.is_file()]
    for qa_file in qa_files:
        try:
            with qa_file.open("r", encoding="utf-8") as f:
                markdown_text = f.read()
                qa_pairs.extend(parse_markdown_to_html(markdown_text))
        except Exception as e:
            print(f"Error reading file {qa_file.name}: {e}")
    return qa_pairs


# ----------------------------------------------------------------------
def create_db() -> None:
    """Create the SQLite database and populate it with questions and answers in HTML format."""

    # app.logger.info(f"{inspect.stack()[0][3]}()")
    with sqlite3.connect(k_DB_Path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS flashcards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question_html TEXT NOT NULL,
            answer_html TEXT NOT NULL
        )
        """
        )
        qa_pairs = load_qa_files(k_QAFolder)  # Parse markdown and convert to HTML before storing
        for qa in qa_pairs:
            cursor.execute(
                "INSERT INTO flashcards (question_html, answer_html) VALUES (?, ?)",
                (qa["question_html"], qa["answer_html"]),
            )
        conn.commit()


# ----------------------------------------------------------------------
# SQLite database setup
def init_db() -> None:
    """Initialize the SQLite database, creating it if it doesn't exist."""

    # app.logger.info(f"{inspect.stack()[0][3]}()")
    if not os.path.exists(k_DB_Path):
        create_db()


# ----------------------------------------------------------------------
def get_random_flashcard(exclude_ids: List[int]) -> Optional[Tuple[int, str, str]]:
    """Get a random flashcard from the database, excluding the ones already seen.

    Args:
        exclude_ids (List[int]): List of flashcard IDs to exclude from selection.

    Returns:
        Optional[Tuple[int, str, str]]: A tuple containing the flashcard ID, question HTML, and answer HTML, or None if no flashcards remain.
    """

    # app.logger.info(f"{inspect.stack()[0][3]}()")
    with sqlite3.connect(k_DB_Path) as conn:
        cursor = conn.cursor()
        # Build query
        if exclude_ids:
            query = "SELECT id, question_html, answer_html FROM flashcards WHERE id NOT IN ({seq}) ORDER BY RANDOM() LIMIT 1".format(
                seq=",".join(["?"] * len(exclude_ids))
            )
            cursor.execute(query, exclude_ids)
        else:
            cursor.execute("SELECT id, question_html, answer_html FROM flashcards ORDER BY RANDOM() LIMIT 1")

        # Fetch the result
        return cursor.fetchone()


# ----------------------------------------------------------------------
# la fonction create_app() est le point d'entrée qui configure l'application Flask avant de la lancer
def create_app() -> Flask:

    logging.basicConfig(level=logging.INFO)

    app = Flask(__name__)
    app.logger.info(f"{inspect.stack()[0][3]}()")
    # If you run the app locally you must run ./secrets.ps1 first (see above)
    # In production on Heroku FLASHCARDS_SECRET_KEY must have been set manually (see readme.md)
    # Without session key, Flask does not allow the app to set or access the session dictionary

    app.secret_key = os.environ.get("FLASHCARDS_SECRET_KEY")

    with app.app_context():
        init_db()  # Initialise la base de données quand l'application est créée

    # Route must be defined inside the create_app otherwise app is not yet defined
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
            current_QA = {"id": flashcard[0], "question_html": flashcard[1], "answer_html": flashcard[2]}
            session["seen_ids"].append(current_QA["id"])  # Add this question to seen list

            return render_template("index.html", Q_html=current_QA["question_html"], A_html=current_QA["answer_html"])
        else:
            return render_template("index.html", Q_html="No more questions.", A_html="")

    # ----------------------------------------------------------------------
    @app.route("/next")
    def next():
        """Route to go to the next question.

        Returns:
            str: Redirect to the index route.
        """

        app.logger.info(f"{inspect.stack()[0][3]}()")
        return redirect(url_for("index"))

    return app


# ----------------------------------------------------------------------
# Application startup
if __name__ == "__main__":

    app = create_app()
    app.logger.info("main()")
    # init_db()  # Initialize or create the database if it doesn't exist
    app.debug = True
    app.run()
