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



<!-- 
############################################################
## 
############################################################ 
-->
Question : Deep Learning - Gradient Descent - If the ``batch size`` is equal to the ``number of observations``, what would the batch gradient descent be equivalent to?
Answer  : 

* It's called batch gradient descent, or simply gradient descent. 
* In this scenario, the algorithm computes the gradient of the cost function with respect to the parameters using the entire dataset at each iteration.


<!-- 
############################################################
## 
############################################################ 
-->
Question : Deep Learning - Gradient Descent - What is the effect of the batch size on the training of the model? 
Answer  : 

* **Computational Efficiency :** Larger batch sizes result in faster training as more samples are processed in parallel.
* **Stability :** Larger batch sizes provide a more stable estimate of the gradient, which can lead to smoother convergence. They might get stuck in local minima more easily.
* **Generalization :** Smaller batch sizes can help the model generalize better as they introduce more randomness in the updates, which can prevent the model from overfitting.


<!-- 
############################################################
## 
############################################################ 
-->
Question : Deep Learning - Gradient Descent - C'est quoi la back propagation?
Answer  : 

* Un algorithme pour entraîner les réseaux de neurones. 
* Il consiste à calculer les gradients des poids du réseau par rapport à une fonction de coût, puis à ajuster ces poids en utilisant un algorithme d'optimisation tel que la descente de gradient, afin de minimiser la perte lors de la phase d'apprentissage. 
* La rétropropagation permet au réseau de s'ajuster progressivement en fonction des erreurs qu'il commet.


<!-- 
############################################################
## 
############################################################ 
-->
Question : Deep Learning - Gradient Descent - C'est quoi Gradient Descent?
Answer  : 

* Une méthode d'optimisation. 
* Minimisation de la dérivée. 
* Minimisation de la fonction de coût en ML. 
* On ne minimise pas toujours la MSE. 
    * MSE légitime en régression linéaire. 
    * En classification on utilisera log loss AKA cross entropy. 
* On trouve alors les valeurs optimales (poids, biais) qui minimisent la loss function. 


<!-- 
############################################################
## 
############################################################ 
-->
Question : Deep Learning - Gradient Descent - Pouvez-vous donner plus de détails sur l'algo du gradient descent?
Answer  :

1. Initialisation (poids, biais)
2. Itération : calculer le grad par rapport au paramètre à optimiser, prendre l'inverse, avancer d'un pas via learning_rate
3. Condition d'arrêt (n_iter)
* La formule : beta(t+1) = beta (t) - gamma * Grad( C )
* Influence de gamma
    * Taille du saut d'un beta au suivant
    * Exploding gradient si trop grand
    * Si trop petit on avance pas


<!-- 
############################################################
## 
############################################################ 
-->
Question : Deep Learning - Gradient Descent - Comment choisir gamma et le nb d'itérations?
Answer  :
    • On trace la fonction de coût en fonction des itérations
    • C doit baisser sur le train set
    • Si C baisse puis augmente => exploding gradient
    • Sur le val set C va baisser puis augmenter => on trouve alors le bon nb d'itérations (overfiting)


<!-- 
############################################################
## 
############################################################ 
-->
Question : Deep Learning - Gradient Descent - What do you say if I say "Batch Gradient Descent, Stochastic Gradient Descent, Mini-Batch Gradient Descent" ? 
Answer  : 

1. **Batch Gradient Descent** (Batch Size = Number of Observations). Full Batch Gradient Descent : In this scenario, the entire dataset is used to compute the gradient of the cost function. The parameters are updated once per epoch (one pass through the entire dataset). This method is computationally expensive because it requires storing the entire dataset in memory and computing the gradients for all samples before updating the parameters. However, it usually leads to very stable updates and can converge to a good solution.
2. **Stochastic Gradient Descent** (Batch Size = 1) : Here, the gradient is computed and parameters are updated after each individual sample. It's very noisy but can help the model escape local minima more easily and often converges faster, especially with large datasets.
3. **Mini-Batch Gradient Descent** (1 < Batch Size < Number of Observations) : A compromise between batch gradient descent and stochastic gradient descent. It computes the gradient and updates the parameters using a subset of the dataset (a mini-batch) at each iteration. This strikes a balance between the stability of batch gradient descent and the faster convergence of stochastic gradient descent. The batch size is typically chosen to be a power of 2 for efficient memory usage.



<!-- 
############################################################
## 
############################################################ 
-->
Question : Deep Learning - Gradient Descent - With batches of 16 observations, how many times will the parameters of the model be updated before we reach one epoch?  
Answer  : 

``N/16``



<!-- 
############################################################
## 
############################################################ 
-->


Question : Deep Learning - Gradient Descent - Peux tu me faire un point sur Batch size et Epochs 
Answer   : 

#### 1. **Batch Size**
Le **batch size** c'est le nombre d'exemples de données sur lesquels le modèle est entraîné avant de mettre à jour les poids via une rétropropagation. C'est la taille des sous-ensembles d'échantillons utilisés pour calculer le gradient et ajuster les poids du modèle.

3 types de traitement basés sur la taille des batchs :

- **Stochastic Gradient Descent (SGD)** : Lorsque la taille du batch est de 1, c’est-à-dire qu’on met à jour les poids après avoir traité chaque exemple de la base de données. Cela peut être bruité, car chaque exemple peut provoquer de grandes variations dans la mise à jour des poids.
- **Mini-batch Gradient Descent** : La taille du batch est un sous-ensemble de l’ensemble de données complet (par exemple, 32, 64, 128). C’est la méthode la plus courante, car elle équilibre la rapidité des mises à jour et la stabilité de la convergence.
- **Batch Gradient Descent** : Ici, le batch size est égal à la totalité de l’ensemble d’entraînement, donc la mise à jour des poids ne se fait qu'une fois par epoch. C’est plus stable, mais plus lent, surtout avec de grands ensembles de données.

#### 2. **Epoch**
Une **epoch** correspond à une passe complète sur l’ensemble de données d’entraînement. C’est le moment où tous les échantillons de données ont été vus une fois par le modèle.

Si on a un data set de 1000 exemples et que le **batch size** est de 100, il faudra 10 batches pour parcourir tous les exemples une fois, ce qui correspondra à **une epoch**.

Pendant l'entraînement, on utilise souvent plusieurs epochs pour permettre au modèle d’apprendre en répétant le processus plusieurs fois. Après chaque epoch, les poids sont ajustés, et le modèle devient progressivement meilleur à mesure que les gradients se raffinent. Bien sûr avant de redemarrer une epoch il faut mélanger (shuffle) le dataset avant de le découper en B batchs.

##### Exemple :
Imaginons qu'on a un dataset de 10 000 exemples, un batch size de 100, et qu'on souhaite entraîner le modèle sur 20 epochs.

- Chaque epoch consistera à traiter 100 batches (10 000 / 100)
- Le modèle verra chaque exemple 20 fois (1 fois par epoch sur 20 epochs), et après chaque batch, il mettra à jour les poids


Le processus est donc :

* Après chaque batch, les poids sont mis à jour en utilisant les gradients calculés pour ce batch.
* Après chaque epoch, aucun ajustement supplémentaire n'est fait, mais on sait qu'on a parcouru tout ton ensemble de données une fois. Si on fait plusieurs epochs, on répète le processus en passant sur tous les exemples plusieurs fois.

Le choix du **batch size** et du nombre d’**epochs** dépend du compromis entre le temps d’entraînement, la qualité des gradients calculés, et la capacité de généralisation du modèle.




<!-- 
############################################################
## 
############################################################ 
-->
Question : Deep Learning - Gradient Descent - Quelle est la différence entre la descente de gradient stochastique (SGD) et la descente de gradient classique ?
Answer  : 

* La descente de gradient stochastique effectue des mises à jour des poids **après chaque exemple** d'entraînement
    * ce qui rend l'optimisation plus rapide mais plus bruitée. 
* La descente de gradient classique calcule les gradients sur **l'ensemble de données** et met à jour les poids une fois
    * ce qui est plus lent mais moins bruité. 
* Garder en tête que si le calcul de grad C est 100 fois plus rapide sur n' points alors on peut se permettre d'avoir une "route" 20 fois moins directe. 
    * Parler du fait que ça peut nous permettre de sortir d'un minimum local.



<!-- 
############################################################
## 
############################################################ 
-->

Question : Deep Learning - Gradient Descent - Peux tu m'expliquer point par point la backpropagation ? Afin de simplifier les choses on se place dans le cas d'un réseau de neurones à 1 neurone et on fait les hypothèses suivantes : 

* Entrée $x_1 = 2$ 
* Poids initial  $w_1 = 0.5 $
* Biais initial $b = 0.0$
* Sortie désirée  $y_{\text{réel}} = 1$ 
* Fonction d'activation sigmoïde : $f(z) = \frac{1}{1 + e^{-z}} $
* La fonction de coût est l'erreur quadratique : $ J = \frac{1}{2} (y_{\text{prévu}} - y_{\text{réel}})^2 $
* Taux d'apprentissage $ \eta = 0.1$


Answer   : 

#### Calcul de la sortie $ y_{\text{prévu}} $

* Somme pondérée : $ z = w_1 \cdot x_1 + b = 0.5 \cdot 2 + 0 = 1.0 $
* Sortie prévisible (à travers la fonction sigmoïde) :
  $$
  y_{\text{prévu}} = f(z) = \frac{1}{1 + e^{-1}} = 0.731
  $$

#### Calcul de la fonction de coût

On injecte $ y_{\text{prévu}} $ et $ y_{\text{réel}} $ dans la fonction de coût.

$$
J = \frac{1}{2} (y_{\text{prévu}} - y_{\text{réel}})^2 = \frac{1}{2} (0.731 - 1)^2 = 0.036
$$

#### Calcul des différentes dérivées partielles pour la rétropropagation

##### Justification :

Dans le cas du poids on va vouloir écrire un truc du style :

$$
w_1 \leftarrow w_1 -\eta \cdot \frac{\partial J}{\partial w_1} = -\eta \cdot \frac{\partial J}{\partial y_{\text{prévu}}} \cdot \frac{\partial y_{\text{prévu}}}{\partial z} \cdot \frac{\partial z}{\partial w_1}
$$

Il faut donc calculer les 3 dérivées partielles suivantes : $\frac{\partial J}{\partial y_{\text{prévu}}}$, $\frac{\partial y_{\text{prévu}}}{\partial z}$ et $\frac{\partial z}{\partial w_1}$

##### Dérivée de la fonction de coût par rapport à la sortie prévisible $ y_{\text{prévu}} $ :

$$
\frac{\partial J}{\partial y_{\text{prévu}}} = \frac{\partial}{\partial y_{\text{prévu}}} \left( \frac{1}{2} (y_{\text{prévu}} - y_{\text{réel}})^2 \right)
$$

$$
\frac{\partial J}{\partial y_{\text{prévu}}} = \frac{1}{2} \cdot 2 \cdot (y_{\text{prévu}} - y_{\text{réel}}) * 1
$$

$$
\frac{\partial J}{\partial y_{\text{prévu}}} = y_{\text{prévu}} - y_{\text{réel}} = 0.731 - 1 = -0.269
$$

##### Dérivée de la sortie prévisible $ y_{\text{prévu}} $ par rapport à $ z $ :



La sortie prévisible est obtenue via la fonction sigmoïde, donc :

$$ y_{\text{prévu}} = f(z) = \frac{1}{1 + e^{-z}} $$

$$
\frac{\partial f(z)}{\partial z} = \frac{-(-e^{-z})}{(1 + e^{-z})^2}  = \frac{e^{-z}}{(1 + e^{-z})^2}
$$

On peut simplifier les choses en remarquant que si

$$ f(z) = \frac{1}{1 + e^{-z}} $$

Alors
 $$ e^{-z} = \frac{1-f(z)}{f(z)} $$

Si on réinjecte dans l'expression de $\frac{\partial f(z)}{\partial z}$ il vient :

$$
\frac{\partial f(z)}{\partial z} = \frac{e^{-z}}{(1 + e^{-z})^2} = \frac{1-f(z)}{f(z)} * \frac{1}{(1 + \frac{1-f(z)}{f(z)})^2}
$$

$$
\frac{\partial f(z)}{\partial z} = (1 - f(z)) \cdot f(z)
$$


$$
\frac{\partial y_{\text{prévu}}}{\partial z} = y_{\text{prévu}} (1 - y_{\text{prévu}}) = 0.731 \cdot (1 - 0.731) = 0.196
$$

##### Dérivée de $z$ par rapport à $w_1$ et $b$ :

On avait :

$$ z = w_1 \cdot x_1 + b $$

Donc : 

* $ \frac{\partial z}{\partial w_1} = x_1 = 2 $
* $ \frac{\partial z}{\partial b} = 1 $

#### Mise à jour des poids et du biais

Les mises à jour se font en tenant compte du taux d'apprentissage $ \eta = 0.1 $.

##### Mise à jour du poids $ w_1 $ :

La variation de poids est : 

$$
\Delta w_1 = -\eta \cdot \frac{\partial J}{\partial w_1} = -\eta \cdot \frac{\partial J}{\partial y_{\text{prévu}}} \cdot \frac{\partial y_{\text{prévu}}}{\partial z} \cdot \frac{\partial z}{\partial w_1}
$$

Avec les valeurs numériques :

$$ \Delta w_1 = -0.1 \cdot (-0.269) \cdot 0.196 \cdot 2 = 0.0105 $$

Le nouveau poids devient :

$$ w_1 \leftarrow w_1 + \Delta w_1 = 0.5 + 0.0105 = 0.5105 $$






##### Mise à jour du biais $ b $ 

La variation du biais est :

$$
\Delta b = -\eta \cdot \frac{\partial J}{\partial b} = -\eta \cdot \frac{\partial J}{\partial y_{\text{prévu}}} \cdot \frac{\partial y_{\text{prévu}}}{\partial z} \cdot \frac{\partial z}{\partial b}
$$

$$
\Delta b = -0.1 \cdot (-0.269) \cdot 0.196 \cdot 1 = 0.0052
$$

Le nouveau biais devient :
$$
b \leftarrow b + \Delta b = 0 + 0.0052 = 0.0052
$$



























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



<!-- 
############################################################
## 
############################################################ 
-->
Question : Deep Learning - Neural networks - Would you say that using neural network models compensates the need for feature engineering?
Answer  : 

* It does. 
* The outputs of the neurons in the network may be interpreted as new features that will be used by later neurons to make even more complex features leading to the final prediciton. 
* In addition, these "features" are build by neurons whose parameters get optimized according to the loss function. 
* So it creates features that are linked to the target variable without having to be explicitely coded. 
* The major **downside** is that it all happens in what may be qualified as a "black box" model.  


<!-- 
############################################################
## 
############################################################ 
-->
Question : Deep Learning - Neural networks - If the model overfits, what can we do to limit overfitting?
Answer  : 

* We can reduce the number of neurons and hidden layers in the network. 
* We can also introduce regularization like Ridge (L2) or Lasso (L1)


<!-- 
############################################################
## 
############################################################ 
-->
Question : Deep Learning - Neural networks - What is the effect of adding neurons on a layer?
Answer  : 

Adding a neuron to a layer makes it possible for the model to create an additional "feature" on a given level of complexity



<!-- 
############################################################
## 
############################################################ 
-->
Question : Deep Learning - Neural networks - What happens if we use a linear activation function? 
Answer  : 

* As a hidden layer 
    * Using a linear activation function is **NOT** a good idea. 
    * We loose the capabilities of neural networks to learn complex relation (non linearities). 
* As an output layer 
    * A linear activation function can be used in regression problems



<!-- 
############################################################
## 
############################################################ 
-->
Question : Deep Learning - Neural networks - What is the effect of adding hidden layers?
Answer  : 

* Adding a hidden layer lets the model add one more level of non-linearity by applying one more activation function to the previous output
* This leads to exponentially complex outputs.


<!-- 
############################################################
## 
############################################################ 
-->
Question : Deep Learning - Neural networks - When you use additional features to feed the model, do you need to use as many neurons and layers? Would adding more neurons and layers be an alternative to using additional features?
Answer  : 

* Adding new features may let you use less complex architectures
* the upside is that you know exactly what input features are used which makes the model more interpretable. 
* On the other hand you may be missing some very useful features that model may have created for you.


<!-- 
############################################################
## 
############################################################ 
-->
Question : Deep Learning - Neural networks - Is it more useful to add more neurons on the layers near the bottom or near the top?
Answer  : 

* It is more useful to add neurons towards the bottom because the complexity of the outputs of earlier neurons limit the complexity of the outputs of later neurons
* It is generally good practice to have more neurons on bottom layers and progressively decrease the number of neurons going up the network.


<!-- 
############################################################
## 
############################################################ 
-->
Question : Deep Learning - Neural networks - Can you list most important activation functions
Answer  : 


<p align="center">
<img src="../static/md/assets/activation.png" alt="activation" width="577"/>
</p>












<!-- https://app.jedha.co/course/introduction-to-tensorflow-ft/01-neural-networks-with-tf-quiz -->


Question : Deep Learning - Neural networks with TensorFlow - What is the purpose of the Dense layer in a neural network?
Answer   : To create a fully connected neuron layer

Question : Deep Learning - Neural networks with TensorFlow - Which layer is used to normalize the output of the preceding layer in a neural network?
Answer   : BatchNormalization layer

Question : Deep Learning - Neural networks with TensorFlow - What is the purpose of the Dropout layer in a neural network?
Answer   : To prevent overfitting

Question : Deep Learning - Neural networks with TensorFlow - Which of the following is an activation function used in neural networks?
Answer   : BatchNormalization, Regularization, Dense, ReLU -> ReLU

Question : Deep Learning - Neural networks with TensorFlow - What is the purpose of regularization in neural networks?
Answer   : To prevent overfitting

Question : Deep Learning - Neural networks with TensorFlow - Which loss function is ideal for most regression problems?
Answer   : MeanSquaredError

Question : Deep Learning - Neural networks with TensorFlow - Which loss function is ideal for binary classification problems?
Answer   : BinaryCrossentropy

Question : Deep Learning - Neural networks with TensorFlow - Which loss function is ideal for multi-class classification problems where the target variable is in dummy form?
Answer   : CategoricalCrossentropy

Question : Deep Learning - Neural networks with TensorFlow - Which loss function is ideal for multi-class classification problems where the target variable is in index form?
Answer   : SparseCategoricalCrossentropy

Question : Deep Learning - Neural networks with TensorFlow - What is the name of the adaptive optimizer that increases or decreases the learning rate based on the gradient value?
Answer   : Adam















































