<!-- 
* Read this : 
* https://www.techwithtim.net/tutorials/flask/sessions


TODO :
* Find a way to NOT send to heroku the whole content of Github
    * Some directories are not necessary while online 
    
* Search engine is quite sensitive
    * "mot-clé" doesn't work because "-"
    * "blablabla()" doesn't work because "()"

* Issues with sessions ?

* Add a links to Chris Albon site under png cards
    * Need to find a way to detect that I'm adding Chris cards
        * Put them in chris_albon directory and treat them (load them) differently ?
    * https://machinelearningflashcards.com/
    * https://fr.slideshare.net/slideshow/machine-learning-flash-cards-by-chris-albonpart2/249940580
    * https://quizlet.com/537961036/chris-albon-ml-flashcards/

* More sample code in 04_fs_big_data.md
* Add CI/CD for automatic testing once testing is working
* Testing 
    * pytest with Flask
    * I don't know yet how to set it up


* ########################################################################
* ########################################################################
* ########################################################################
* DONE - Add questions about books to read ?
    * Should I create a non-technical category ?
* DONE - Add non technical question about job interview ?    

* DONE - Document how to add grahics in a directory close to the .md file 
    * For example ./static/md/my_directory/one_topic.md contains one question which display the image in 
    * .static/md/my_directory/assets/graph1.png
    * Then it MUST use
    ```
    <p align="center">
    <img src="../static/md/dummy_test/assets/dummy.png" alt="dummy" width="577"/>
    </p>
    ```
    * Indeed, the html page "is" in .\py-flashcards\templates\index.html and the path to the resource must take it inot account
    * Is there a way to use kinfd of "url_for("blablabla") ??????

* DONE - Add a favicon
* DONE - Add support for cards based on png images
    * png cards are available here : https://app.jedha.co/course/technical-interviews-part-2-ft/practice-machine-learning-ft
    * Load the paths to png files in a list
    * insert it into the answer field
    * insert de keyword in the question filed such that we know this is NOT a Q&A but just an illusion... an image
    * in order to display
        * check if it is an image (check the content of the question)
        * if image display it
        * otherwise display Q&A as usual
* DONE - Allow .md or .png to be in sub-directories under ./static/md and ./static/png respectively 
* DONE - Fix logger issues such that any function can log properly
    * => add and configure a global logger
* DONE - Create an SQL code snippet .md file
* DONE - Create an sns code snippet .md file
* DONE - Create a matplotlib code snippet .md file
* DONE - Add serch engine (see /search route)
* DONE - Display nb of cards in search
* DONE - Display the total number of cards somewhere
* DONE - Add an EDA code snippet .md file


-->

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
$env:FLASHCARDS_SECRET_KEY = "blablabla"
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
    * heroku config:set FLASHCARDS_SECRET_KEY=blablabla 
    * heroku open
    * This should work

# Workflow
## To run locally
* Open a terminal in VSCode and run ``./secrets.ps1`` once
    * You can type ``ls env:FLASH*`` to double check
* Modify files etc.
* Optional - Commit on github from VSCode    
* Strike F5 while ``py-flashcards.py`` is open
    * If the app complains
        1. Stop the app (CTRL+C)
        1. The Python Debug Console should be opened
        1. Run ``./secrets.ps1`` once in the Python Debug Console

## To deploy on Heroku
* Modify files etc.
* Commit on github from VSCode    
* ``git push heroku main``
* type ``heroku open`` in the terminal (or visit the app web page)

# Q&A

---
* Q : How to check gunicorn is serving the app on Heroku?
* A : Open a terminal locally
    * heroku logs --tail
    * CTRL+C 
    * CTRL+F gunicorn
    * You should see a line similar to : `[INFO] Starting gunicorn 23.0.0`

---
* Q : Can I organize the markdown files in directories and sub-directories ?
* A : Yes as long as they are under the ``./static/md`` directory 

---
* Q : Can I organize the .png cards in directories and sub-directories ?
* A : Yes as long as they are under the ``./static/png`` directory 

---
* Q : Can I insert a link to a .png file or a link to a web page into the answer ?
* A : Yes. Answers are plain markdown files so you can insert 
    * link to images
    * source code
    * bold, italic fonts
    * equations and math symbols (Latex syntax)


---
* Q : The answer includes a link to a ``.png`` file. What is the path I should use in the content of the answer ? 
* A : Let's take an example :
    1. Your markdown file (which includes one or more set of question/answer) is in ``.\static\md\dummy_test`` directory
    1. The name of the markdown file is ``dummy.md``
    1. Underneath ``dummy.md`` there is an ``assets`` where there is `dummy.png`
    
To make it clear, so far, the organization of the files looks like :

```
./md
│   other_QA_file_01.md
│   other_QA_file_02.md
│
└───dummy_test
    │   dummy.md
    │
    └───assets
            dummy.png
```

Here is how to point the ``./md/dummy_test/assets/dummy.png`` from the file ``./md/dummy_test/dummy.md``.

```
Question : The question I want to ask

Answer  : 
This is the expected answer with a link to the ``./assets/dummy.png`` 


<p align="center">
<img src="../static/md/dummy_test/assets/dummy.png" alt="dummy" width="577"/>
</p>

```
* Keep in mind you **MUST** point the ``dummy.png`` as if you were in ``./templates/index.html``
* Indeed the markdown text of the questions and answers is inserted into the ``index.html``
 


# About contributions
This project was developed for personal and educational purposes. Feel free to explore and use it to enhance your own learning in machine learning.

Given the nature of the project, external contributions are not actively sought or encouraged. However, constructive feedback aimed at improving the project (in terms of speed, accuracy, comprehensiveness, etc.) is welcome. Please note that this project was created as part of a certification process, and it is unlikely to be maintained after the final presentation.    
