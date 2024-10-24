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
from pathlib import Path

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Add your own key and put in place a robust protection

# k_PNGFolder = os.path.join("static", "png_cards")
# k_Img_Folder = Path("static") / "png_cards"
k_PNGFolder = "./static/png"

# Obtenir la liste de toutes les images
# images = os.listdir(k_PNGFolder)
# images = [file for file in Path(k_PNGFolder).iterdir() if file.is_file()]
images = [file for file in Path(k_PNGFolder).rglob("*.png") if file.is_file()]
images = [Path(*file.parts[2:]) for file in images]
images = [str(file).replace("\\", "/") for file in images]


# img_folder = Path(k_Img_Folder)
# images = list(img_folder.iterdir())
# images = list(k_Img_Folder.iterdir())
print(images)


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

    return render_template("index_for_test_02_img.html", img_file=img_file)


@app.route("/next")
def next_image():
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
