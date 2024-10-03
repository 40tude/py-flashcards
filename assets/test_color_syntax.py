from markdown import markdown

# Exemple de texte en Markdown avec des extensions
content = """
# Ceci est un titre

Voici une équation mathématique :

$$ E = mc^2 $$

Et un peu de code en Python :

```python
def hello_world():
    print("Hello, World!")
```
"""

# Test de l'extension extra
try:
    content_html_extra = markdown(content, extensions=["extra"])
    print("L'extension extra fonctionne correctement.")
except Exception as e:
    print("Erreur avec l'extension extra : ", str(e))

# Test de l'extension codehilite
try:
    content_html_codehilite = markdown(content, extensions=["codehilite"])
    print("L'extension codehilite fonctionne correctement.")
except Exception as e:
    print("Erreur avec l'extension codehilite : ", str(e))
