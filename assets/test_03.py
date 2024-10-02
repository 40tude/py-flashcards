# python test_03.py or F5

from flask import Flask, render_template, redirect, url_for, session
from markdown import markdown
import re
from pathlib import Path
import random

app = Flask(__name__)

k_QAFolder = "./static/md"
# qa_pairs: list[str] = []
qa_pairs: list[str] = []


# -----------------------------------------------------------------------------
def parse_markdown(markdown_text):
    pattern = re.compile(r"\* Question : (.*?)\* Réponse\s*: (.*?)(?=\n\* Question|\Z)", re.DOTALL)
    matches = pattern.findall(markdown_text)
    return [
        {"question": "**Question : **" + match[0].strip(), "answer": "**Answer : **" + match[1].strip()}
        for match in matches
    ]


# -----------------------------------------------------------------------------
def load_qa_files(directory):
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


# -----------------------------------------------------------------------------
@app.route("/")
def index():
    if "unseen_QA" not in session or not session["unseen_QA"]:
        session["unseen_QA"] = qa_pairs.copy()
        session["seen_QA"] = 0

    if session["unseen_QA"]:
        current_QA = random.choice(session["unseen_QA"])
        session["unseen_QA"].remove(current_QA)
        session["seen_QA"] += 1

        Q_html = markdown(current_QA["question"], extensions=["extra", "codehilite", "sane_lists"])
        A_html = markdown(current_QA["answer"], extensions=["extra", "codehilite", "sane_lists"])

        return render_template("index_for_test_03.html", Q_html=Q_html, A_html=A_html)
    else:
        return render_template("index_for_test_03.html", Q_html="No more questions.", A_html="")


# -----------------------------------------------------------------------------
@app.route("/next")
def next():
    return redirect(url_for("index"))


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app.secret_key = "supersecretkey"

    qa_pairs = load_qa_files(k_QAFolder)
    app.debug = True
    app.run()
