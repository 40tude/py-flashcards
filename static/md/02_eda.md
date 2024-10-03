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

Question : NUMPY - What is the result of matrix multiplication in NumPy?
Answer  : A new matrix where each element is the dot product of rows and columns

Question : NUMPY - How are eigenvalues and eigenvectors of a matrix in NumPy computed?
Answer  : 
```python
np.linalg.eig()
```
Question : NUMPY - What does the "@" symbol represent in NumPy in linear algebra context?
Answer  : Matrix multiplication

Question : NUMPY - What is slicing in NumPy?
Answer  : A way to access a portion of an array or matrix

Question : NUMPY - How do you create an array of zeros in NumPy?
Answer  : 
```python
np.zeros()
```

Question : NUMPY - How do you change the shape of an existing NumPy array?
Answer  : 
```python
reshape()
```

Question : NUMPY - What is a NumPy mask?
Answer  : An array of boolean values indicating where a condition is met

Question : NUMPY - What is an example of discrete quantitative data?
Answer  : Nb people in a room

Question : STAT101 - Which measure of central tendency is the most affected by outliers?
Answer  : Mean

Question : STAT101 - What does a Z-score measure?
Answer  : The number of standard deviations a data point is from the mean

Question : STAT101 - Which sampling method should be avoided due to potential bias?
Answer  : Convenience Sampling (vs Random Sampling Stratified Sampling  Cluster Sampling)

Question : PANDAS - How do you read a CSV file using Pandas?
Answer  : 
```python
pd.read_csv(filename)
```
Question : PANDAS - What are the two primary data structures in Pandas?
Answer  : DataFrames and Series

Question : PANDAS - How do you convert a string to a datetime object in Pandas?
Answer  : 
```python
pd.to_datetime() 
pd.to_datetime(df['dates'], format='%Y-%m-%d')
…
```

Question : PANDAS - What is the purpose of the .dt accessor in Pandas?
Answer  : To access datetime properties and methods on Series

Question : PANDAS - How can you extract the year from a datetime object in Pandas?
Answer  : 
```python
datetime_object.year
```

Question : PANDAS - What does the Timedelta type represent in Pandas?
Answer  : The difference between two datetime values

Question : DATAVIZ - What is the primary purpose of data visualization?
Answer  : To make complex data easy to understand

Question : DATAVIZ - Which of the following is a key principle in creating effective data visualizations?
Answer  : Keeping a consistent style and scaling

Question : DATAVIZ - How does color affect data visualization?
Answer  : Color can help highlight key information

Question : DATAVIZ - What does a good data visualization aim to achieve? 
Answer  : Convey complex information in an easy-to-understand format

Question : DATAVIZ - What is an important aspect to consider when choosing a color scale for your visualization?
Answer  : How the colors can accurately represent and differentiate the data

Question : PLOTLY - What is Plotly Express primarily used for?
Answer  : Generating a variety of static and interactive charts

Question : PLOTLY - How do you display a chart in a Jupyter notebook using Plotly Express?
Answer  : Using the ``fig.show()`` method after creating the chart

Question : PLOTLY - What type of data visualization can be created with Plotly Express?
Answer  : A wide range of chart types, including line, bar, scatter, and more

Question : PLOTLY - Is it possible to customize the appearance of charts in Plotly Express?
Answer  : Yes, including aspects like color, layout, and annotations

Question : PLOTLY - What is a primary advantage of using Plotly Express for data visualization?
Answer  : It offers a simple syntax for quickly creating a variety of charts


<!-- 
https://app.jedha.co/course/interactive-graphs-ft/quiz-fs-m03-v1

What is pandas and its purpose?

What are the two main object types from pandas?

What method would you use if you had to quickly compute usual descriptive statistics on the DataFrame "df" taking into account both numerical and non-numerical features?

What is numpy, and what's its purpose?

What is the main object type in numpy?

It's possible to easily convert pandas.DataFrame objects into numpy objects, what attributes of the pandas class can be used for this?

What is the seaborn method you would use to plot the distribution of a variable?

What is the purpose of using this command before running some visualization code?

What is the module that lets you make easy interactive graph in plotly and which one is the more complicated version of the module?

What is the plotly method that lets you visualize your data points as dots in a two-dimensional space?

Where is the best place to learn?
-->