# Locally :
#   you must define the session key before to run/debug the app
#       ./secrets.ps1
#       ls env:FLASH*                 # to double check
#   python py-flashcards.py or F5
#   flask --app py-flashcards run

import logging
from flask import Flask, render_template, session, redirect, url_for
from markdown import markdown
import re
from pathlib import Path
import random
import os

logging.basicConfig(level=logging.INFO)  # set logging level. Msg>= info are recorded
app = Flask(__name__)

# If you run the app locally you must run ./secrets.ps1 first (see above)
# In production on Heroku FLASHCARDS_SECRET_KEY must have been set manually (see readme.md)
# Without session key, Flask does not allow the app to set or access the session dictionary
app.secret_key = os.environ.get("FLASHCARDS_SECRET_KEY")
# print(app.secret_key)


# -----------------------------------------------------------------------------
# TODO : The order & position of this function matters => FIXIT
def parse_markdown(markdown_text: str) -> list:
    pattern = re.compile(r"\* Question : (.*?)\* Réponse\s*: (.*?)(?=\n\* Question|\Z)", re.DOTALL)
    matches = pattern.findall(markdown_text)
    return [
        {"question": "**Question : **" + match[0].strip(), "answer": "**Answer : **" + match[1].strip()}
        for match in matches
    ]


# -----------------------------------------------------------------------------
# TODO : The order & position of this function matters => FIXIT
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
    # print(qa_pairs)
    return qa_pairs


k_QAFolder = "./static/md"
qa_pairs = load_qa_files(k_QAFolder)


# -----------------------------------------------------------------------------
@app.route("/")
def index() -> str:
    # debug, info, warning, error, critical
    app.logger.info("access to home page")

    if "unseen_QA" not in session or not session["unseen_QA"]:
        session["unseen_QA"] = qa_pairs.copy()
        session["seen_QA"] = 0

    if session["unseen_QA"]:
        current_QA = random.choice(session["unseen_QA"])
        session["unseen_QA"].remove(current_QA)
        session["seen_QA"] += 1

        Q_html = markdown(current_QA["question"], extensions=["extra", "codehilite", "sane_lists"])
        A_html = markdown(current_QA["answer"], extensions=["extra", "codehilite", "sane_lists"])

        return render_template("index.html", Q_html=Q_html, A_html=A_html)
    else:
        return render_template("index.html", Q_html="No more questions.", A_html="")


# -----------------------------------------------------------------------------
@app.route("/next")
def next():
    return redirect(url_for("index"))


# -----------------------------------------------------------------------------
# This code is NOT executed while on Heroku
if __name__ == "__main__":
    app.debug = True
    app.run()
