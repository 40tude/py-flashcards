# flask --app test_01 run

from flask import Flask, render_template, url_for
from markdown import markdown

app = Flask(__name__)


@app.route("/")
def index():
    image1_url = url_for("static", filename="png_cards/Accuracy_web.png")
    content = f"""
# Ceci est un titre

Voici une équation mathématique :

$$ E = mc^2 $$

Et un peu de texte en **gras** et en *italique*.

Une image :

![Accuracy]({image1_url})

Du code Python :

```python
def hello_world():
    print("Hello, World!")
```

Du code HTML :

```html
<body>
    <div>
        <p>Hello</p>
    </div>
</body>
```



"""

    # Convertir le markdown en HTML
    content_html = markdown(content, extensions=["extra", "codehilite"])
    return render_template("index_for_test_01.html", content=content_html)


if __name__ == "__main__":
    app.run(debug=True)
