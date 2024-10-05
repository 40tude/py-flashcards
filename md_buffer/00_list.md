<!-- Entretien

	• https://sites.google.com/view/datascience-cheat-sheets
	
	
	
Relire toutes les question et ajouter "- **" à la fin si elle est importante
	• Se limiter OBLIGATOIREMENT à 20% des questions
	
Faire le quiz 
	• LIN-REG Quiz: Maximum likelihood estimation 
	• LIN-REG Quiz: How to compute metrics 
	• Time Series : test your knowledge!
	• asynchronous programming

############################################################
## Questions à traiter, ranger plus tard
############################################################


EDA	                    : The recipe
Features Engineering	: The secret sauce
Baseline model	          : The first taste
Metrics Analysis	     : The critics' score
API & App	               : Sharing with friends
Deployment Monitoring	: Serve the dish, maintain quality




Transformer, any comment ?
The attention mechanism that learns contextual relationships between words in a text 


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



#### Code snippet 

```python
# TODO : add sample code
```

-->


<!-- 

############################################################
## Divers
############################################################ 

-->



Question : NLP - What is the purpose of oov_token in ``tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=k_num_words, oov_token="_UNKNOWN_")``
Answer  : 
``oov_token`` = out of vocabulary token
1. When the oov_token is specified in the tokenizer, words not present in the learned vocabulary will be replaced by this token. 
This enables the model to handle new words that appear in test or inference data, while reducing the risk of errors or inaccuracies.
In a new sentence, 2 unknown words will be represented by the OOV token, preserving information even if the model hasn't seen these words before
2. If oov_token not specified any word not in the vocabulary will be ignored and not tokenized
This may result in the loss of important information during inference or testing.





Question : Classes déséquilibrées, spam par exemple. Vous faites quoi ?
Answer  : 
	1. train_test_split() avec stratify
	2. Touche plus au jeu de test
	3. Equilibrer les classes du train set (50/50)
	4. Entrainer le modèle avec le train set équilibré
	5. Validation/métriques avec le jeu de test déséquilibré 

L'équilibre des classes se fait par sous ou sur-échantillonnage. 
Sur échantillonnage  : imblearn ou SMOTE (Synthetic Minority Oversampling Technique, synthèse de points)
Sous échantillonnage : Tomek Links ou NearMiss
Si on veut garder des classes déséquilibrées lors du training on peut faire de la pondération de classe
C'est l'inverse de la freq des classes
Voir class_weight de sklearn qui retourne un dictionnaire qu'on passe ensuite à model.fit() de tensorflow (param class_weight)
On peut aussi faire du sample_weight où chaque échantillon de y_train recoit une pondération spécifique à sa classe 
Voir param sample_weight de model.fit() de tensorflow
Attention aux metrics : conf matrix, precision, recall,F1 score, area under ROC curve
Lire : https://www.analyticsvidhya.com/blog/2020/07/10-techniques-to-deal-with-class-imbalance-in-machine-learning/


Question : MSE, RMSE, MAE, R², MAPE ?
Answer  :
RMSE généralement préférée dans les pb de régression
Si outliers, faut peut-être prendre en compte la MAE
RMSE (L2 based) est plus sensible aux outliers que MAE (L1 based)
MAE et RMSE sont 2 mesures de distance entre vecteurs (prédictions et target)
Différentes mesures de distances sont possibles:
RMSE : norme euclidienne, L2
MAE : norme de Manhattan, L1
Plus le n de Ln augmente et plus la norme focalise sur les grandes valeurs en négligeant les petites
Quand les outliers sont exponentiellement rares (bell curve) RMSE est plus efficace

MAPE : 
Erreur Absolue Moyenne en % de la vraie valeur
Exprimée en %, c'est une mesure simple et intuitive de l'accuracy d'un modèle
Important de l'utiliser avec prudence quand les valeurs vraies sont proches de zéro.
MAPE=0% : modèle parfait. 
MAPE élevé : erreurs importantes dans les prédictions


Question : ANOVA?
Answer  : Analyse de la variance. La variation c'est l'information. ANOVA c'est analyser la quantité d'information captée par le modèle. Variance=écart à la moyenne.

Question : Silhouette, un commentaire? 
Answer  : Le coefficient de silhouette évalue la cohésion et la séparation des clusters. Paramètre global du clustering. On veut des clusters bien regroupés et bien séparés. Coef sans unité. Entre -1 et 1. On veut 1. 0.5 OK. On choisit k tel que s soit maximal (voir aussi analyse courbe WCSS, Elbow). Le score de silhouette est calculé pour chaque point de données en mesurant la similarité avec son propre cluster par rapport aux clusters voisins les plus proches.


Question : Data Warehouse vs Databases ?
Answer  : 
Data warehouses : optimized to have a performance boost on columns (features)
Databases       : optimized for extracting rows (observations)

Question : What is data tidying?
Answer  : this is the process of transforming raw, messy data into a clean and organized format that is easier to analyze and interpret. This process involves structuring the data in a way that :
Each variable forms a column
Each observation forms a row
Each type of observational unit forms a table


Question : What is Spark SQL 
Answer  : It brings native support for SQL to Spark. It streamlines the process of querying data stored both in RDDs (resilient distributed datset, Spark’s distributed datasets) and in external sources. Spark SQL blurs the lines between RDDs and relational tables. Unifying these abstractions makes it easy for developers to intermix SQL commands querying external data with complex analytics, all within in a single application.

Question : What are the types of categorical data ?
Answer  : Ordinal Data & Nominal Data
Ordinal data is a ranking list. It’s ordered, BUT the intervals between the ranks aren’t necessarily equal. 
Nominal data is like choosing your favorite ice cream flavor. There’s no logical order to the choices. Whether it’s “Vanilla,” “Chocolate,” or “Strawberry,” one isn’t inherently better or worse than the others. 

Question : D'où provient le terme recall ?
Answer  : Le terme "recall" en machine learning vient du domaine de la récupération d'information (Information Retrieval). Dans ce contexte, "recall" se réfère à la capacité d'un système à retrouver toutes les occurrences pertinentes dans un ensemble de données. 
En d'autres termes, le "recall" mesure le pourcentage de vrais positifs parmi tous les éléments pertinents. Il est utilisé pour évaluer la performance des modèles de classification, en particulier dans les situations où il est crucial de ne pas manquer des éléments pertinents (par exemple, dans la détection de maladies, où il est important de ne pas manquer de cas positifs). 

Question : RND-FOREST - Boostrapping ?  
Answer  : On crée plusieurs sous-ensembles de données en échantillonnant de manière aléatoire avec remplacement à partir de l'ensemble de données d'origine. Chaque sous-ensemble peut donc contenir des exemples répétés et ne pas inclure certains exemples de l'ensemble d'origine.

Question : RND-FOREST - Bagging ?
Answer  : Parallèle (indépendance).Le bagging consiste à entraîner plusieurs modèles indépendamment les uns des autres sur différentes versions d'un même ensemble de données, puis à combiner leurs prédictions pour obtenir un modèle final plus robuste. 3 phases : Bootstrap Sampling, Entraînement des modèles, Agrégation des résultats (classification => agrégation par vote majoritaire. Régression => agrégation par la moyenne des prédictions). 
Objectif : Réduction de la variance. Exemple : Random Forest

Question : BOOST-ADABOOST - Boosting ?
Answer  : Séquentiel. Le boosting consiste à entraîner plusieurs modèles de manière séquentielle, chaque modèle cherchant à corriger les erreurs des modèles précédents. Les modèles sont construits de manière dépendante et on pondère les exemples d'entraînement en fonction des erreurs des modèles précédents. 
Objectif : Réduction du biais et de la variance (correction des erreurs progressives). Exemple : XGBoost, AdaBoost
3 phases : 
Initialisation          : Un premier modèle est entraîné sur l'ensemble de données d'origine.
Ajustement              : Les exemples mal classés ou mal prédits par le modèle précédent sont pondérés davantage, de sorte que le modèle suivant se concentre sur ces erreurs.
Combinaison des modèles : Les modèles sont combinés en pondérant leurs prédictions en fonction de leur performance. En général, les modèles performants reçoivent un poids plus important.




Question : XGBoost ?
Answer  :
Extreme Grandient Boosting
based on decision trees
several weak models (decision trees) are combined to form a robust model
Model 1 predictions are compared with true values
Each model is trained to minimize a loss function that measures the residual error
Residuals are kept and become the target values of the next model
The new set of weighted observations is injected into model 2
At the end we have n models
Submit an observation.
Sum of the n predictions (each tree predict a residual)

Question : XGBoost benefits drawbacks for predictions
Answer  :
### Benefits
- **Handles complex and heterogeneous data**: XGBoost works well with diverse features (numerical, categorical) and can model complex non-linear relationships often found in price prediction tasks.
- **High performance with large datasets**: It is designed to be highly efficient, even with large volumes of data, thanks to parallelization and memory optimization.
- **Manages missing values and outliers**: XGBoost automatically handles missing data and can incorporate outliers without negatively impacting performance, which is crucial for price prediction scenarios.
- **Prevents overfitting**: With regularization techniques, depth control, and learning rate adjustments, XGBoost helps reduce overfitting, making it suitable for price models with high complexity.
- **Feature importance and interpretability**: XGBoost provides insights into feature importance, helping identify key factors that influence price predictions, which is valuable for decision-making.
- **Robust and adaptable**: It performs well across different data distributions and can handle cases where price variance is high, adapting to various relationships between features and target variables.


### Potential Drawbacks
- **Complex hyperparameter tuning**: To achieve optimal performance, XGBoost often requires careful hyperparameter tuning, which can increase development complexity.
- **Training time**: While fast, training can be longer on very large datasets compared to simpler algorithms, though the test performance typically makes up for this. 




Question : LOG-REG - Pouvez-vous m'expliquer les termes "précision" et "accuracy"
Answer  : Métriques utilisées pour évaluer la performance des modèles de classification.
	• L'accuracy (exactitude) est une mesure globale de la performance d'un modèle. C'est le pourcentage de prédictions correctes sur le total des prédictions : (TP + TN)/(total de prédictions)
	• Précision : une mesure la qualité des prédictions positives (précision, positive). C'est le pourcentage de prédictions positives correctes par rapport au nombre total de prédictions positives. TP/(TP+FP)
	• L'accuracy donne une idée globale de la performance du modèle, mais peut être trompeuse si les classes sont déséquilibrées. La précision est cruciale lorsque les faux positifs ont un coût élevé, comme dans le dépistage des maladies, où un faux positif pourrait entraîner des tests supplémentaires inutiles.

Question : LOG-REG - C'est quoi le F1 score, que signifie-t-il, quand l'utiliser?
Answer  : Moyenne harmonique du recall et de la précision. Utile quand 1 - les classes sont déséquilibrées 2 - Si les faux positifs et les faux négatifs ont des coûts comparables, le F1 score fournit un bon compromis. Si proche de 1 =>  le modèle a une bonne performance, équilibrant  précision et rappel. Si faible => soit le modèle a une faible précision, soit un faible rappel, soit les deux. Le modèle ne performe pas bien et nécessite des ajustements.

Question : REG-LIN - Forward selection, Backward selection
Answer  : 
	• Forward selection  : on ajoute les variables au modèle qui à chaque étape augment le R². On arrête si y a plus variable ou si R² baisse.
	• Backward selection : Elimination. On part avec toutes les variables. On élimine la variable qui a la plus forte probabilité de ne pas être pertinante (p-value). On arrête quand toutes les variables ont une p-value sup à 5%

Question : SVM - C'est quoi un kernel ? 
Answer  : Function take the observations into a larger dimensional space, in which we hope that the geometric properties of the observations will be linearly separable

Question : Objectif de K-Means ?
Answer  : L'objectif du K-Means est de regrouper des données en K clusters de telle manière que les points à l'intérieur d'un cluster soient similaires entre eux et différents des points d'autres clusters, en minimisant la variance intra-cluster. La variance intra-cluster est une mesure de la dispersion des données à l'intérieur de chaque cluster. Elle représente la somme des carrés des distances entre chaque point de données d'un cluster et le centre de ce cluster.


Question : K-Means - WCSS… Any comment ? 
Answer  : Within Cluster Squared Sum. Voir méthode ELBOW (Densité des clusters). Pour chaque exécution de K-Means, on calcule la WCSS (within cluster squared sum, somme des distances au carré entre chaque point de données et le centroïde de son cluster correspondant). C'est un paramètre global sur l'ensemble des clusters. C'est la somme des sommes des carré

Question : REG-LIN - homoscédasticité ? 
Answer  : On parle d'homoscédasticité lorsque la variance des erreurs stochastiques de la régression est la même pour chaque observation i (de 1 à n observations). 

Question : REG-LIN - Quelles sont les 3 hypothèses que l'on fait en régression linéaire
Answer : 
	• Linéarité : évident
	• Indépendance des erreurs : l'erreur sur une observation est indépendante de l'erreur sur une autre. Difficile à prouver à partir d'échantillons.Corrélation vs Causation
	• Homoscédasticité : La distribution des erreurs est indépendante de y. Faut que la distribution des erreurs, que l'écart à la droite soit constant qqsoit y

Question : TBD - What is Boto3 the SDK (Software Development Kit) of AWS?
Answer  : It is a collection of tools and libraries to help you use AWS using code

Question : REG-LIN - What is a RDBMS (Relational DataBase Management System)?
Answer  : A piece of software that lets define, create, maintain, and control access to a database







Question : C'est quoi le machine learning ? 
Answer  : Machine Learning (ML) = sous-domaine de l'IA 
Se concentre sur le développement de techniques permettant aux ordinateurs d'apprendre à partir de données et d'améliorer leurs performances sans être explicitement programmés pour chaque tâche. 
Le ML permet aux systèmes informatiques :  
	1. de reconnaître des modèles dans les données 
	2. de faire des prédictions ou de prendre des décisions basées sur ces modèles
	3. sans intervention humaine directe pour spécifier explicitement les règles.


Question : Expliquez ce qu'est la validation croisée et pourquoi c'est important dans le contexte de l'apprentissage automatique ?
Answer  : Technique utilisée pour évaluer les performances d'un modèle en divisant les données en sous-ensembles d'apprentissage et de test de manière itérative. Cela permet d'estimer la capacité de généralisation du modèle sur des données non vues, et d'identifier le surapprentissage. Les méthodes courantes incluent la validation croisée en k-fold et la validation croisée leave-one-out.

Question : Neural Network. What is the effect of adding neurons on a layer?
Answer  : Adding a neuron to a layer makes it possible for the model to create an additional "feature" on a given level of complexity

Question : Différence entre régression et classification en apprentissage automatique ?
Answer  : La régression est utilisée pour prédire une valeur continue, tandis que la classification est utilisée pour prédire une classe ou une catégorie discrète.

Question : Neural Network. If the model overfits, what can we do to limit overfitting?
Answer  : We can reduce the number of neurons and hidden layers in the network. We can also introduce regularization like Ridge (L2) or Lasso (L1)

Question : Quelle est l'importance de l'analyse exploratoire des données (EDA) dans le processus de modélisation en science des données ?
Answere : L'analyse exploratoire des données permet de comprendre la structure, les modèles et les relations présentes dans les données avant de construire des modèles. Cela inclut l'examen des distributions, des corrélations, des valeurs manquantes, et des outliers, ce qui guide les décisions sur le nettoyage des données et le choix des modèles. Penser à parler aussi des distributions qui sont skewed d'un côté ou de l'autre (1.5 IQR vs 3 sigma). Parler aussi du déséquilibre dans la target.

Question : Pouvez-vous expliquer ce qu'est l'overfitting et comment le détecter ?
Answer  : L'overfitting se produit lorsque le modèle s'adapte trop précisément aux données d'entraînement et perd sa capacité de généralisation sur de nouvelles données. Il peut être détecté en observant une performance élevée sur les données d'entraînement mais une performance médiocre sur les données de test, ou en comparant les performances du modèle sur les données d'entraînement et de validation.

Question : With batches of 16 observations, how many times will the parameters of the model be updated before we reach one epoch?  
Answer  : N/16

Question : Quelle est la différence entre la normalisation et la standardisation des données ?
Answer  : 
La normalisation   met à l'échelle les données dans une plage spécifique, souvent entre 0 et 1. 
La standardisation transforme les données pour qu'elles aient une moyenne nulle et un écart-type de 1.

Question : Neural Network. What happens if we use a linear activation function? 
Answer  : Using a linear activation function as a hidden layer is not a good idea. We loose the capabilities of neural networks to learn complex relation (non linearities). As a output layer it can be used in regression problems

Question : Qu'est-ce qu'une fonction de coût (ou de perte) et comment est-elle utilisée dans l'apprentissage automatique ?
Answer  : Une fonction de coût mesure l'erreur entre les prédictions d'un modèle et les valeurs réelles de l'ensemble de données. Elle est utilisée dans le processus d'optimisation pour guider l'ajustement des paramètres du modèle afin de minimiser cette erreur. Parler de régression => MSE Claasification => LogLoss (entropy)

Question : What is the effect of adding hidden layers?
Answer  : Adding a hidden layer lets the model add one more level of non-linearity by applying one more activation function to the previous output, leading to exponentially complex outputs.

Question : Pouvez-vous expliquer ce qu'est la réduction de la dimensionnalité et pourquoi est-ce important dans l'analyse de données ?
Answer  : La PCA consiste à réduire le nombre de variables ou de caractéristiques dans un ensemble de données. Cela permet de simplifier les modèles, de réduire le temps de calcul et de prévenir le surapprentissage, tout en préservant autant que possible les informations importantes.

Question : Neural Network. When you use additional features to feed the model, do you need to use as many neurons and layers? Would adding more neurons and layers be an alternative to using additional features?
Answer  : Adding new features may let you use less complex architectures, the upside is that you know exactly what input features are used which makes the model more interpretable. On the other hand you may be missing some very useful features that model may have created for you.

Question :Quelles sont les différences entre l'apprentissage supervisé et l'apprentissage non supervisé ?
Answer  : L'apprentissage supervisé implique l'utilisation de données étiquetées pour entraîner un modèle à prédire une sortie, tandis que l'apprentissage non supervisé explore les données pour découvrir des structures intrinsèques sans étiquettes. Parler des cas d'usage du non supervisé. Pas une fin en soi

Question : Is it more useful to add more neurons on the layers near the bottom or near the top?
Answer  : It is more useful to add neurons towards the bottom because the complexity of the outputs of earlier neurons limit the complexity of the outputs of later neurons. It is generally good practice to have more neurons on bottom layers and progressively decrease the number of neurons going up the network.

Question : Quelle est la différence entre la descente de gradient stochastique (SGD) et la descente de gradient classique ?
Answer  : La descente de gradient stochastique effectue des mises à jour des poids après chaque exemple d'entraînement, ce qui rend l'optimisation plus rapide et plus bruitée. La descente de gradient classique calcule les gradients sur l'ensemble de données et met à jour les poids une fois, ce qui est plus lent mais moins bruité. 

Si le calcul de grad C est 100 fois plus rapide sur n' points on peut se permettre d'avoir une "route" 20 fois moins directe. Parler du fait que ça peut nous permettre de sortir d'un minimum local.

Question : Would you say that using neural network models compensates the need for feature engineering?
Answer  : It does. The outputs of the neurons in the network may be interpreted as new features that will be used by later neurons to make even more complex features leading to the final prediciton. In addition, these "features" are build by neurons whose parameters get optimized according to the loss function. So it creates features that are linked to the target variable without having to be explicitely coded. The major downside is that it all happens in what may be qualified as a "black box" model.  

Question : Expliquez ce qu'est la régularisation Lasso et en quoi elle diffère de la régularisation Ridge ?
Answer  : La régularisation Lasso ajoute, à la fonction de coût, une pénalité proportionnelle à la valeur absolue des coefficients du modèle, ce qui favorise la sélection de caractéristiques importantes et conduit à une certaine sparsité. La régularisation Ridge utilise une pénalité proportionnelle au carré des coefficients, ce qui réduit la magnitude des coefficients sans les éliminer complètement.

Question : If the batch size is equal to the number of observations, what would the batch gradient descent be equivalent to?
Answer  : It's called batch gradient descent, or simply gradient descent. In this scenario, the algorithm computes the gradient of the cost function with respect to the parameters using the entire dataset at each iteration.

Question : C'est quoi la régularisation ?
Answer  : Une technique pour réduire le surapprentissage (overfitting) en pénalisant les modèles trop complexes. Consiste à ajouter un terme de pénalité à la fonction de coût lors de l'entraînement. Ca encourage le modèle à privilégier des solutions plus simples. La régularisation aide à améliorer la généralisation du modèle en contrôlant sa complexité et en réduisant le risque de surapprentissage.

Question : What is the effect of the batch size on the training of the model? 
Answer  : It includes
	• Computational Efficiency : Larger batch sizes result in faster training as more samples are processed in parallel.
	• Stability : Larger batch sizes provide a more stable estimate of the gradient, which can lead to smoother convergence. They might get stuck in local minima more easily.
	• Generalization : Smaller batch sizes can help the model generalize better as they introduce more randomness in the updates, which can prevent the model from overfitting.

Question : C'est quoi la back propagation?
Answer  : Un algorithme pour entraîner les réseaux de neurones. Il consiste à calculer les gradients des poids du réseau par rapport à une fonction de coût, puis à ajuster ces poids en utilisant un algorithme d'optimisation tel que la descente de gradient, afin de minimiser la perte lors de la phase d'apprentissage. La rétropropagation permet au réseau de s'ajuster progressivement en fonction des erreurs qu'il commet.

Question : C'est quoi Gradient Descent?
Answer  : Une méthode d'optimisation. Minimisation de la dérivée. Minimisation de la fonction de coût en ML. On ne minimise pas toujours la MSE. MSE légitime en régression linéaire. En classification on utilisera log loss AKA cross entropy. On trouve alors les valeurs optimales (poids, biais) qui minimisent la loss function. 

Question : Pouvez-vous donner plus de détails sur l'algo du gradient descent?
Answer  :
	1. Initialisation (poids, biais)
	2. Itération : calculer le grad par rapport au paramètre à optimiser, prendre l'inverse, avancer d'un pas via learning_rate
	3. Condition d'arrêt (n_iter)
La formule : beta(t+1) = beta (t) - gamma * Grad( C )
Influence de gamma : taille du saut d'un beta au suivant. Exploding gradient si trop grand. Si trop petit on avance pas

Question : Elbow method, any comment?
Answer  : Dans le contexte de l'algorithme K-Means (non supervisé), la méthode Elbow est utilisée pour déterminer le nombre optimal de clusters à utiliser. 
1. Exécution de K-Means pour différents nombres de clusters (de 2 à 20 par exemple). 
2. Calcul de la variance intra-cluster (inertie) : Pour chaque exécution de K-Means, on calcule la WCSS (within cluster squared sum, somme des distances au carré entre chaque point de données et le centroïde de son cluster correspondant). Mesure de la dispersion des données à l'intérieur de chaque cluster. L'inertie intra-cluster. Plus le nombre de clusters est élevé, plus l'inertie intra-cluster tend à diminuer. WCSS para global à l'ensemble des clusters. La somme des sommes des carrés
3. Tracé du graphique inertie intra-cluster vs nombre de clusters. 
4. Identification du point de coude sur le graphe où recherche le point où la décroissance de l'inertie intra-cluster commence à ralentir de manière significative. C'est le point où ajouter un cluster de plus k=k+1 ne fait pas basser WCSS de manière significative. Cela ressemble à un coude sur le graphique. Ce point est souvent considéré comme le nombre optimal de clusters à utiliser. Dans certains cas, le point de coude peut ne pas être clairement défini. Il peut être utile alors d'utiliser d'autres méthodes de validation des clusters (Silhouette).

Question : Compromis biais-variance
Answer  : 
1. Le biais mesure à quel point les prédictions d'un modèle diffèrent des valeurs réelles. Un modèle avec un biais élevé simplifie trop les données d'entraînement et sous-estime la complexité de la relation entre les features et la target. Conduit à des performances médiocres sur les données d'entraînement et de test. Les modèles à haut biais sont généralement trop simples pour capturer la complexité des données. Pour réduire le biais, on peut utiliser des modèles plus complexes ou augmenter la taille ou la complexité des caractéristiques utilisées. Features engineering.
2. La variance mesure la sensibilité d'un modèle aux petites variations dans l'ensemble de données d'entraînement. Un modèle avec une variance élevée est trop sensible au bruit dans les données d'entraînement, ce qui peut conduire à un surajustement. Le modèle fonctionne bien sur les données d'entraînement mais il a du mal à généraliser sur de nouvelles données. Les modèles à haute variance sont souvent complexes (arbres de décision profonds, réseaux neuronaux avec de nombreux paramètres). Pour réduire la variance, on peut utiliser la régularisation, la réduction de la dimensionnalité ou l'augmentation des données.
La validation croisée peut être utile pour évaluer comment le compromis biais-variance affecte les performances du modèle. En utilisant la validation croisée, on peut ajuster les hyperparamètres du modèle pour trouver le meilleur compromis entre biais et variance.

Question : Batch Gradient Descent, Stochastic Gradient Descent, Mini-Batch Gradient Descent ? 
Answer  : 
1. Batch Gradient Descent (Batch Size = Number of Observations). Full Batch Gradient Descent. In this scenario, the entire dataset is used to compute the gradient of the cost function. The parameters are updated once per epoch (one pass through the entire dataset). This method is computationally expensive because it requires storing the entire dataset in memory and computing the gradients for all samples before updating the parameters. However, it usually leads to very stable updates and can converge to a good solution.
2. Stochastic Gradient Descent (Batch Size = 1). Here, the gradient is computed and parameters are updated after each individual sample. It's very noisy but can help the model escape local minima more easily and often converges faster, especially with large datasets.
3. Mini-Batch Gradient Descent (1 < Batch Size < Number of Observations). A compromise between batch gradient descent and stochastic gradient descent. It computes the gradient and updates the parameters using a subset of the dataset (a mini-batch) at each iteration. This strikes a balance between the stability of batch gradient descent and the faster convergence of stochastic gradient descent. The batch size is typically chosen to be a power of 2 for efficient memory usage.
