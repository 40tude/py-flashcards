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


<!-- https://app.jedha.co/course/gradient-descent-course-ft/01-gradient-descent-quiz -->

Question : Deep Learning - Gradient Descent - What is the goal of the gradient descent algorithm?
Answer   : To find the set of parameters that minimizes the loss function

Question : Deep Learning - Gradient Descent - What is a loss function?
Answer   : A function that measures how bad the model's prediction errors are

Question : Deep Learning - Gradient Descent - What does the gradient of a function represent?
Answer   : The vector indicating the direction of greatest increase at a given point

Question : Deep Learning - Gradient Descent - What is stochastic gradient descent?
Answer   : A type of gradient descent that uses a batch of samples for each update

Question : Deep Learning - Gradient Descent - Why is a grid search not a suitable method for optimizing model parameters?
Answer   : It requires a large amount of computational power

Question : Deep Learning - Gradient Descent - Which step of the gradient descent algorithm modifies the model parameters iteratively to decrease the loss function?
Answer   : Iteration

Question : Deep Learning - Gradient Descent - What does the learning rate determine in the gradient descent algorithm?
Answer   : The speed at which the parameters are updated

Question : Deep Learning - Gradient Descent - What is the most common stopping criterion in deep learning?
Answer   : Limiting the number of gradient descent steps

Question : Deep Learning - Gradient Descent - What happens if the learning rate is too small in gradient descent?
Answer   : The algorithm converges slowly or may not converge at all

Question : Deep Learning - Gradient Descent - What is the "explosive gradient problem"?
Answer   : When the learning rate is too high, causing the loss function to increase uncontrollably

Question : Deep Learning - Gradient Descent - What is the main difference between gradient descent and stochastic gradient descent?
Answer   : Gradient descent uses all training samples to compute the gradient, while stochastic gradient descent uses a random subset of samples

Question : Deep Learning - Gradient Descent - What is an epoch in the context of training models?
Answer   : The unit of measurement to track model training progress, representing one pass through the entire training dataset






<!-- https://app.jedha.co/course/introduction-to-neural-networks-ft/01-neural-networks-quiz -->


Question : Deep Learning - Neural networks - What is the purpose of an activation function in a neural network? 
Answer   : To add non-linearity to the network's behavior

Question : Deep Learning - Neural networks - Which activation function is known for being computationally efficient and allowing for backpropagation? 
Answer   : ReLu

Question : Deep Learning - Neural networks - What is the disadvantage of the ReLu activation function? 
Answer   : It is not zero-centered

Question : Deep Learning - Neural networks - What type of architecture is organized in a sequential manner, with input, hidden, and output layers? 
Answer   : Sequential architecture

Question : Deep Learning - Neural networks -  What is the purpose of the forward pass in a neural network?
Answer   : To transform inputs into outputs

Question : Deep Learning - Neural networks -  How are layers connected in a sequential neural network architecture?
Answer   : Each layer receives inputs from the previous layer

Question : Deep Learning - Neural networks -  What are hyperparameters in a neural network?
Answer   : The neural network model architecture

Question : Deep Learning - Neural networks - How do data scientists typically determine the number of layers and neurons in a neural network? 
Answer   : Through trial and error and comparing models

Question : Deep Learning - Neural networks - What is the purpose of an activation function in a neural network? 
Answer   : To add non-linearity to the network's behavior
































