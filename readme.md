### ATTENTION
* Heroku does not allow "_" in project name
* Use "-" to name your project instead

# Note
* I use WIN11 (while Heroku runs Linux) but most of the information are platform independent
* I use conda

# How to
You should have all the files already. The lines below explain how the project was initially set up.
* conda create --name py-flashcards python=3.12 -y
* conda activate py-flashcards
* create directory py-flashcards 
* cd ./py-flashcards 
* code .
* create file mypy.ini
* create file py-flashcards.py
* conda install flask mypy markdown pygments -y
* create a secrets.ps1 similar to

```
$env:FLASHCARDS_SECRET_KEY = "<blablabla...>"
```
* create .gitignore
    * at least, add a line with : secrets.ps1
* Open a terminal in VSCode (CTRL + ù)
    * ./secrets.ps1
* Strike F5 in VScode
    * The app should be running locally
    * CTRL+C
* conda list -e > ./assets/requirements_conda.txt
* pip list --format=freeze > requirements.txt
    * At the end of requirements.txt manually add the line "gunicorn==23.0.0"
    * I have to do that because I run WIN11 and I can't install gunicorn
    * gunicorn is only used in "production" on heroku
    * If you run Linux
        * conda install gunicorn -y
        * pip list --format=freeze > requirements.txt
* create file Procfile
    * Pay attention to :  py-flashcards:create_app()
    * name of the Python file + ":" + entry_point()
    * in py-flashcards.py take a look at create_app()
        * Gunicorn uses the create_app() function to obtain the Flask application instance, and starts the WSGI server
* create file runtime.txt
* From VSCode commit to github
* From the VSCode integrated terminal 
    * heroku login
    * heroku create py-flashcards
        * https://py-flashcards-41b349ab0591.herokuapp.com/ 
        * https://git.heroku.com/py-flashcards.git
        * are created for example
    * git remote add heroku https://git.heroku.com/py-flashcards.git
    * git push heroku main
    * heroku config:set FLASK_ENV=production
    * heroku config:set FLASHCARDS_SECRET_KEY=<blablabla> 
    * heroku open
    * This should work

# Workflow
## To run locally
* Open a terminal in VSCode and run ``./secrets.ps1`` once
    * You can type ``ls env:FLASH*`` to double check
* Modify files etc.
* Commit on github from VSCode    
* Strike F5 while ``py-flashcards.py`` is open

## To deploy on Heroku
* Modify files etc.
* Commit on github from VSCode    
* ``git push heroku main``
* type ``heroku open`` in the terminal (or visit the app web page)

# Q&A
* Q : How to check gunicorn is serving the app on Heroku?
* A : Open a terminal locally
    * heroku logs --tail
    * CTRL+C 
    * CTRL+F gunicorn
    * You should see a line similar to : `[INFO] Starting gunicorn 23.0.0`

# About contributions
This project was developed for personal and educational purposes. Feel free to explore and use it to enhance your own learning in machine learning.

Given the nature of the project, external contributions are not actively sought or encouraged. However, constructive feedback aimed at improving the project (in terms of speed, accuracy, comprehensiveness, etc.) is welcome. Please note that this project was created as part of a certification process, and it is unlikely to be maintained after the final presentation.    
