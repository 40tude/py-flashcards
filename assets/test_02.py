# move this file above the static dir
# test_01.py
# |
# ───static
# │   ├───md
# │   └───png_cards
# │           Accuracy_web.png
# │           ...

# uncomment the line app.secret...
# flask --app test_02 run

from flask import Flask, render_template, session, redirect, url_for
import os
import random

app = Flask(__name__)
# app.secret_key = "supersecretkey"  # Add your own key and put in place a robust protection

IMG_FOLDER = os.path.join("static", "png_cards")

# Obtenir la liste de toutes les images
images = os.listdir(IMG_FOLDER)


@app.route("/")
def index():
    if "seen_images" not in session:
        session["seen_images"] = []

    if len(session["seen_images"]) == len(images):
        # Toutes les images ont été vues, réinitialiser la liste
        session["seen_images"] = []

    remaining_images = list(set(images) - set(session["seen_images"]))
    img_file = random.choice(remaining_images)
    session["seen_images"].append(img_file)

    return render_template("index_for_test_02.html", img_file=img_file)


@app.route("/next")
def next_image():
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
