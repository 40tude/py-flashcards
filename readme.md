


* **ATTENTION** heroku do not allow "_" in project name
    * Use "-" instead
* conda create --name py-flashcards python=3.12 -y
* conda activate py-flashcards
* create directory py-flashcards
* code .
* create file mypy.ini
* create file py-flashcards.py
* conda install flask mypy markdown pygments -y
* create a secrets.ps1 similar to
```
$env:FLASHCARDS_SECRET_KEY = "blablabla..."
```
* create .gitignore
    * at least, add a line with : secrets.ps1
* Strike F5 in VScode
    * Should be running locally
    * CTRL+C
* From VSCode commit on github 
* conda list -e > ./assets/requirements_conda.txt
* pip list --format=freeze > requirements.txt
    * At the end of requirements.txt manually add the line "gunicorn==23.0.0"
    * I have to do that because I run WIN11 and I can't install gunicorn
    * gunicorn is only used in "production" on heroku
* create file Procfile
    * Pay attention to :  py-flashcards:app
    * name of the .py file, :, app
* create file runtime.txt
* From VSCode commit to github
* VSCode integrated terminal (CTRL + ù)
    * heroku login
    * heroku create py-flashcards
        * https://py-flashcards-41b349ab0591.herokuapp.com/ 
        * https://git.heroku.com/py-flashcards.git
        * are created
    * git remote add heroku https://git.heroku.com/py-flashcards.git
    * git push heroku main
    * heroku config:set FLASK_ENV=production
    * heroku config:set FLASHCARDS_SECRET_KEY=blablabla 
    * heroku open
    * This should work

* Workflow
    * Modify files etc.
    * Commit on github from vscode    
    * git push heroku main
    * heroku open (or visit the app web page)

* Q : How to check gunicorn works is serving the app
* A : 
    * integrated terminal
    * heroku logs --tail
    * CTRL+C
    * CTRL+F gunicorn
    * You should see : [INFO] Starting gunicorn 23.0.0
