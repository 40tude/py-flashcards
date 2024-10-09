import logging
from flask import Flask, render_template, session, redirect, url_for
from markdown import markdown
import re
from pathlib import Path
import random
import os
import sqlite3

logging.basicConfig(level=logging.INFO)  # set logging level. Msg>= info are recorded
app = Flask(__name__)

app.secret_key = os.environ.get("FLASHCARDS_SECRET_KEY")

# Database file path
DB_PATH = "./flashcards.db"


# ----------------------------------------------------------------------
# SQLite database setup
def init_db():
    if not os.path.exists(DB_PATH):
        create_db()


# Create a new SQLite database and populate it with data from Markdown files
def create_db():
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
        qa_pairs = load_qa_files(k_QAFolder)
        for qa in qa_pairs:
            cursor.execute("INSERT INTO flashcards (question, answer) VALUES (?, ?)", (qa["question"], qa["answer"]))
        conn.commit()


# Get a random flashcard from the database that hasn't been shown yet
def get_random_flashcard(exclude_ids):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        query = "SELECT id, question, answer FROM flashcards WHERE id NOT IN ({seq}) ORDER BY RANDOM() LIMIT 1".format(
            seq=",".join(["?"] * len(exclude_ids)) if exclude_ids else "SELECT id FROM flashcards"
        )
        cursor.execute(query, exclude_ids)
        return cursor.fetchone()


# ----------------------------------------------------------------------
# Parse the markdown files to extract question-answer pairs
def parse_markdown(markdown_text: str) -> list:
    markdown_text = re.sub(r"<!--.*?-->", "", markdown_text, flags=re.DOTALL)
    pattern = re.compile(r"Question\s*:\s*(.*?)\nAnswer\s*:\s*(.*?)(?=\nQuestion|\Z)", re.DOTALL)
    matches = pattern.findall(markdown_text)
    return [
        {"question": "###Question :\n" + match[0].strip(), "answer": "###Answer :\n" + match[1].strip()}
        for match in matches
    ]


# Load the Markdown files from a directory
def load_qa_files(directory: str) -> list:
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
# Flask routes
@app.route("/")
def index():
    app.logger.info("Accessing home page")

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


@app.route("/next")
def next():
    return redirect(url_for("index"))


# ----------------------------------------------------------------------
# Application startup
if __name__ == "__main__":
    init_db()  # Initialize or create the database if it doesn't exist
    app.debug = True
    app.run()
