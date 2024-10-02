# flask --app test_00 run
# or F5 in VSCode (from there you can set breakpoint...)

from flask import Flask

app = Flask(__name__)


@app.route("/")
def hello_world():
    return "<p>Hello, World!!!!</p>"


if __name__ == "__main__":
    app.run(debug=True)
