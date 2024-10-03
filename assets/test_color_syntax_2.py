from flask import Flask, render_template
from markdown import markdown

app = Flask(__name__)


@app.route("/")
def index() -> str:

    content = """
# This is a title
## Sub title

```python
def hello_world():
    print("Hello, World!")
```    
"""

    # content_html = markdown(content, extensions=["extra", "codehilite"])
    content_html = markdown(content, extensions=["extra", "codehilite"], output_format="html5")
    print(content_html)
    return render_template("index_for_color-syntax.html", content=content_html)


if __name__ == "__main__":
    app.run(debug=True)


# if __name__ == "__main__":

#     content_for_test = """
# ```python
# def hello_world():
#     print("Hello, World!")
# ```
# """

#     try:
#         content_html_extra = markdown(content_for_test, extensions=["extra"])
#         print("L'extension extra fonctionne correctement.")
#         print(content_html_extra)
#     except Exception as e:
#         print("Erreur avec l'extension extra : ", str(e))

#     try:
#         content_html_codehilite = markdown(content_for_test, extensions=["codehilite"])
#         print("L'extension codehilite fonctionne correctement.")
#         print(content_html_extra)
#     except Exception as e:
#         print("Erreur avec l'extension codehilite : ", str(e))

#     app.run(debug=True)
