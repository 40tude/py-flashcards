<!-- 
08 10 2024
DONE : Les quizz de la section Python ont été vu

-->




<!-- 
<p align="center">
<img src="../static/md/assets/img1.png" alt="attention" width="577"/>
</p>

$$ E = mc^2 $$




#### Code snippet  

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


Question : 
Answer   : 

#### Code snippet 

```python
# TODO : add sample code
```

-->



<!-- 
############################################################
## Questions issues des quizz
############################################################ 
-->

Question : PYTHON - Data types - Give 2 examples of mutable data collection?
Answer  : List, Dictionary



Question : PYTHON - Data types - How do you add a single element to the end of a list? How can you remove an item from a list using its value?
Answer  : 

#### Code snippet 

```python
append()
delete()

```

#### Code snippet 

```python
list_villes=["Aix en Provence", "Paris", "Ghisonaccia"]
urls=[]
for ville in list_villes:
    urls.append(
        f"https://www.booking.com/searchresults.fr.html?ss={ville}&checkin=2024-05-02&checkout=2024-05-05&group_adults=2&no_rooms=1&group_children=0",
    )

```



Question : PYTHON - Data types - What is the syntax for creating a slice of a list that includes elements from index 2 to index 5 (excluding index 5)?
Answer  : 

#### Code snippet 

```python
list[2:5]
```





Question : PYTHON - Data types - How can you access the value associated with a specific key in a dictionary?
Answer  : Use brackets or .get(). Pay attention when the key does'nt exist yet.

#### Code snippet 

```python
you = {'name': 'Zoubida', 'age': 42}

print(f'First name : {you.get('name')}')
print(f"Age        : {you['age']}")
```




Question : PYTHON - Data types - How do you add a new key-value pair to a dictionary?
Answer  : 

#### Code snippet 

```python
dico["bob"] = 42
```




Question : PYTHON - Data types - How can you iterate over both keys and values in a dictionary?
Answer  : 

#### Code snippet 

```python
person = {
  "nom" : "ROBERT",
  "prenom" : "Zoubida"
}
print(person["nom"])
person.items()
```




Question : PYTHON - Functions - How can you add a default argument to a function?
Answer  : By assigning a value to the parameter (param="bob") in the function declaration.

#### Code snippet 

```python
def volume(length=1, width=1, depth=1):
  print(f"Length = {length}")
  return length * width * depth;

volume(42, 2, 3)
volume()
volume(width=4)
```




Question : PYTHON - Functions - What does the acronym DRY stand for in programming?
Answer   : Don't repeat yourself



Question : PYTHON - Functions - What is the purpose of giving an alias to exceptions?
Answer  : To customize the error message displayed to the user.



Question : PYTHON - Functions - How can you create your own exception in Python?
Answer  : By using the ``raise`` statement with a specific error message.

#### Code snippet 

```python
def find_seat(self, n):
    if (not isinstance(n, int) or n < 0):
        raise Exception("n should be a positive integer")
```




Question : PYTHON - Classes - Which method is used to initialize the attributes of a class in Python?
Answer  : 

#### Code snippet 

```python
class Employee():
 
  # Initializing
  def __init__(self, a_name):
    print('Employee created.')
    self.name = a_name

  # Deleting (Calling destructor)
  def __del__(self):
    print('Destructor called, Employee deleted.')
    self.name=""
```




Question : PYTHON - Classes - What does the ``self`` keyword represent in Python classes?
Answer  : It refers to the instance of the class.

#### Code snippet 

```python
class MyImputer():
  
  def __init__(self, mylist:list[int]):
    tmp_list = []
    for i in range(len(mylist)):
      if (mylist[i] != "None"):
        tmp_list.append(mylist[i])
    
    avg = sum(tmp_list)/len(tmp_list)
    
    self.list = mylist.copy()
    for i in range(len(self.list)):
      if (self.list[i] == "None"):
        self.list[i] = avg

  def display(self):
    print(self.list)  
```


Question : PYTHON - Classes - What does the ``ValueError`` exception indicate?
Answer  : It is raised when a method is called with incorrect arguments. 

#### Code snippet 

```python
class MyCustomImputer(BaseEstimator, TransformerMixin):

    def __init__(self, strategy='mean'):
        self.strategy = strategy

    def fit(self, X, y=None):
        if self.strategy == 'mean':
            self.fill_value = np.nanmean(X, axis=0)
        elif self.strategy == 'median':
            self.fill_value = np.nanmedian(X, axis=0)
        elif self.strategy == 'most_frequent':
            self.fill_value = np.nanmax(X, axis=0)
        else:
            raise ValueError("Invalid strategy. Please choose 'mean', 'median', or 'most_frequent'.")
        return self

    def transform(self, X):
        return np.where(np.isnan(X), self.fill_value, X)
```        


<!-- 
############################################################
## Questions qui ne proviennent pas des quizz
############################################################ 
-->

Question : PYTHON - Différence entre **arguments** et **paramètres**
Answer  : 
* Les **paramètres** d'une fonction sont les noms listés dans la définition de la fonction. 
* Les **arguments** d'une fonction sont les valeurs passées à la fonction.





Question : PYTHON - Pourquoi voudriez-vous implémenter la méthode ``__call__()`` dans la classe d'un de vos modèles?
Answer  : If the model class has a ``__call__()`` method then we can call it as a function. 
 
#### Code snippet 

```python
print("model output:", my_model(data))
```

