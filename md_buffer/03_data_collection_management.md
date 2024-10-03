<!-- 
<p align="center">
<img src="../static/md/assets/img1.png" alt="attention" width="577"/>
</p>

$$ E = mc^2 $$

```python
# -----------------------------------------------------------------------------
def preprocessor(df):
    # drop
    df.drop(columns="Unnamed: 7", inplace=True)
    df.drop_duplicates(inplace=True)

    # format
    df.columns = df.columns.str.lower()
    df.columns = df.columns.str.replace("/", "_")
```

-->



<!-- 
############################################################
## Questions issues des quizz
############################################################ 
-->

Question : HTTP - What is the purpose of the HTTP protocol?
Answer  : To transfer data and instructions over the Internet

Question : HTTP - What does an HTTP request consist of? 
Answer  : Method, URL, and protocol version

Question : url et route ?
Answer  : Dans https://api.github.com/zen 

* L'**url** c'est https://api.github.com
* La **route** c'est zen  

Question : HTTP - Which HTTP method is used to request a resource at a specified URL?
Answer  : ``GET``

Question : HTTP - What does an HTTP response contain?
Answer  : Status code, headers, and body

Question : HTTP - What does a status code of 404 indicate?
Answer  : Resource is no longer available at the requested location

Question : HTTP - Which response header provides information about the type of content in the body of the response?
Answer  : Content-Type

Question : HTTP - Which status code indicates a client error?
Answer  : ``401``

Question : HTTP - What is the difference between a ``GET`` and a ``POST`` method for HTTP requests?
Answer  : 

* The **POST** method lets you send data to the web server
* While the **GET** method only gathers data from the web server without sending any





Question : API - Which HTTP method is used to retrieve data from an API?
Answer  : ``GET``

Question : API - What does REST stand for in REST API?
Answer  : 
**Representational State Transfer**. Décrit une architecture où : 

* les interactions avec des ressources web passent par des échanges de représentations de ces ressources. On ne manipule pas les ressources mais leur repésentation.
* l'état de l'application (les données) est transféré à chaque requête de manière stateless (pas de session en mémoire d'une requête à une autre)



Question : API - Which Python library can be used to interact with APIs?
Answer  : requests

Question : API - How can you add parameters to a ``GET`` request?
Answer  : Use the params parameter

```python
my_params = {
  "q" : "paris",
  "countrycodes" : "fr",
  "format":"json",
}
response = requests.get(url, params=my_params) 
```

Question : API - Which HTTP method is used to send data to an API?
Answer  : ``POST``

Question : API - How can you access the content of a response as plain text in requests library?
Answer  : ``response.text``

Question : API - How can you retrieve binary content, such as an image, from an API response?
Answer  : ``response.content``

Question : HTML-CSS - Which CSS selector is used to select an element by its class?
Answer  : ``.class``





Question : SCRAPY - How can you rotate user agents in Scrapy?
Answer  : By installing the ``scrapy-user-agents`` library and configuring the ``settings.py`` file.

Question : SCRAPY - How can you specify a list of rotating proxies in Scrapy?
Answer  : By installing the ``scrapy-rotating-proxies`` library and configuring the ``settings.py`` file.

Question : SCRAPY - Which command is used to start a Scrapy spider contained in a project?
Answer  : 
```bash
scrapy crawl spider_name
```

Question : SCRAPY - What is the purpose of the ``.follow()`` method in Scrapy?
Answer  : To navigate to the next page in a pagination sequence.

Question : SCRAPY - What is the purpose of the Scrapy AutoThrottle extension?
Answer  : To automatically adjust Scrapy to the optimum crawling speed and avoid exceeding requests limitations.

Question : SCRAPY - Which Scrapy commands allows you to start a new Scrapy project?
Answer  : 
```bash
scrapy startproject
```
Question : SCRAPY - What is Scrapy used for?
Answer  : Parsing HTML pages, Scraping websites automatically, Running multiple crawlers simultaneously

Question : SCRAPY - What does the parse() method in a Scrapy spider do?
Answer  : It defines the callback function for processing the response


Question : SCRAPY - How can you avoid being banned from websites when using Scrapy?
Answer  : 
1. Use a different IP address for each request
1. Slow down the crawling speed
1. Randomize the order of requests


Question : SCRAPY - What is the purpose of the CrawlerProcess in Scrapy?
Answer  : It sets up the user agent for scraping

Question : SCRAPY - How can you save the results of a Scrapy spider in a JSON file?
Answer  : Specify the output file using the FEEDS setting

Question : SCRAPY - What are callbacks used for in Scrapy?
Answer  : To perform tasks that are independent of the code itself

Question : SCRAPY - How can you navigate the web and follow links using Scrapy? 
Answer  : By using the `.follow()` method and providing the XPath of the link

Question : SCRAPY - How can you authenticate on a website using Scrapy?
Answer  : By using the ``.from_response()`` method and sending a ``POST`` request with the login data.

Question : SCRAPY - What is the purpose of Scrapy projects?
Answer  : To configure the scraping process and manage settings.

Question : SCRAPY - How can you enable AutoThrottle in Scrapy?
Answer  : By uncommenting the appropriate lines in the settings.py file.

```python
# settings.py

BOT_NAME = 'myproject'

SPIDER_MODULES = ['myproject.spiders']
NEWSPIDER_MODULE = 'myproject.spiders'

# Enable AutoThrottle
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 5
AUTOTHROTTLE_MAX_DELAY = 60
AUTOTHROTTLE_TARGET_CONCURRENCY = 1.0
AUTOTHROTTLE_DEBUG = False

# Other settings ...
```







Question : ETL - What does ETL stand for?
Answer  : Extract Transform Load

Question : ETL - What is the purpose of an ETL process?
Answer  : To clean and load data into a database

Question : ETL - In the context of ETL, what is the role of Extract?
Answer  : To gather data from various sources

Question : ETL - Which storage system is commonly used as a datalake?
Answer  : AWS S3

Question : ETL - What is the purpose of transforming data in an ETL process?
Answer  : To clean and validate data

Question : ETL - Why is an ETL process useful for business intelligence (BI)?
Answer  : To gather data in one place for analysis

Question : ETL - What is a Data Warehouse?
Answer  : A database specifically optimized for analytics

Question : ETL - What does ETL process ensure for data in a company?
Answer  : Data accuracy and validity

Question : ETL - When would a company greatly benefit from implementing an ETL process?
Answer  : When it has multiple data sources

Question : ETL - What is the primary purpose of a transactional database and a data warehouse, respectively?
Answer  : A transactional database is primarily used for day-to-day operational tasks, while a data warehouse is primarily used for historical data analysis

Question : SQL - What is a relational database?
Answer  : A database consisting of 2-dimensional tables

Question : SQL - What is a DBMS?
Answer  : Database Management System

Question : SQL - What does a schema represent in a database?
Answer  : The structure of the database

Question : SQL - What is a line in a table also known as?
Answer  : Record

Question : SQL - What does NULL represent in SQL?
Answer  : Unknown or missing values

Question : SQL-ALCHEMY - What is SQLAlchemy?
Answer  : Python library for manipulating databases

Question : SQL-ALCHEMY - Which layer of SQLAlchemy allows you to communicate with databases and create flexible models?
Answer  : ORM

Question : SQL-ALCHEMY - What is the purpose of the ``__repr__`` method in SQLAlchemy? 
Answer  : It formats the output of an object

Question : SQL-ALCHEMY - How do you persist values in a database using SQLAlchemy?
Answer  : By calling the ``commit()`` method

Question : SQL-ALCHEMY - How can you query data from a database using SQLAlchemy?
Answer  : By using the ``query()`` function
