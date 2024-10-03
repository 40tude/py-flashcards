import re
from markdown import markdown


# def parse_markdown(markdown_text: str) -> list:
#     # Regex pour capturer les paires question-réponse
#     pattern = re.compile(r"Question\s*:\s*(.*?)\nAnswer\s*:\s*(.*?)(?=\nQuestion|\Z)", re.DOTALL)
#     matches = pattern.findall(markdown_text)

#     # Formatage des résultats dans une liste de dictionnaires
#     return [{"question": match[0].strip(), "answer": match[1].strip()} for match in matches]


def parse_markdown(markdown_text: str) -> list:
    markdown_text = re.sub(r"<!--.*?-->", "", markdown_text, flags=re.DOTALL)
    # pattern = re.compile(r"\* Question\s*:\s*(.*?)\nAnswer\s*:\s*(.*?)(?=\n\* Question|\Z)", re.DOTALL)
    pattern = re.compile(r"Question\s*:\s*(.*?)\nAnswer\s*:\s*(.*?)(?=\nQuestion|\Z)", re.DOTALL)
    matches = pattern.findall(markdown_text)
    return [
        {"question": "**Question : **" + match[0].strip(), "answer": "**Answer : **" + match[1].strip()}
        for match in matches
    ]


# Exemple avec le nouveau texte markdown
markdown_text = """

<!-- 

Question : Q1
Answer  : A1

-->

Question: PYTHON - Différence entre argument et paramètre
Answer: 
* Les paramètres d'une fonction sont les noms listés dans la définition de la fonction. 
* Les arguments d'une fonction sont les valeurs passées à la fonction.
"""

# Test de la fonction
result = parse_markdown(markdown_text)
for item in result:
    # print(item)
    print(item["question"])
    print(item["answer"])
    print("*****************")

    Q_html = markdown(item["question"], extensions=["extra", "codehilite", "sane_lists"])
    print(Q_html)

    A_html = markdown(item["answer"], extensions=["extra", "codehilite", "sane_lists"])
    print(A_html)
