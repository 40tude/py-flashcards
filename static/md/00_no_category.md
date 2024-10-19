<!-- Entretien

	• https://sites.google.com/view/datascience-cheat-sheets
	
	
	
Relire toutes les question et ajouter "<je sais pas pquoi encore>" à la fin si elle est importante
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


Question : 
Answer   : 

#### Code snippet 

```python
# TODO : add sample code
```

-->


<!-- 
############################################################
## 
############################################################ 
-->
Question : No category yet - What is the purpose of oov_token in ``tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=k_num_words, oov_token="_UNKNOWN_")``
Answer  : 

* ``oov_token`` = out of vocabulary token
* When the ``oov_token`` is specified in the tokenizer, words which are **NOT** present in the learned vocabulary will be replaced by this token. 
    * This enables the model to handle new words that appear in test or inference data, while reducing the risk of errors or inaccuracies.
    * In a new sentence, 2 unknown words will be represented by the OOV token, preserving information even if the model hasn't seen these words before
* If ``oov_token`` not specified any word not in the vocabulary will be ignored and not tokenized
    * This may result in the loss of important information during inference or testing.




<!-- 
############################################################
## 
############################################################ 
-->
Question : No category yet - Classes déséquilibrées, spam par exemple. Vous faites quoi ?
Answer  : 

1. ``train_test_split()`` avec stratify
1. On touche plus au jeu de test
1. Equilibrer les classes du train set (50/50)
1. Entrainer le modèle avec le train set équilibré
1. Validation/métriques avec le jeu de test déséquilibré 

L'équilibrage des classes se fait par sous ou sur-échantillonnage

* Sur échantillonnage
    * RandomOverSampler from ``imblearn.over_sampling``  
    * SMOTE (Synthetic Minority Oversampling Technique, synthèse de points)

<p align="center">
<img src="../static/md/assets/smote.png" alt="smote" width="577"/>
</p>

* Sous échantillonnage 
    * Tomek Links 
    * NearMiss
* Si on veut garder des classes déséquilibrées lors du training on peut faire de la pondération de classe
    * C'est l'inverse de la freq des classes
    * Voir ``class_weight`` de sklearn qui retourne un dictionnaire qu'on passe ensuite à ``model.fit()`` de tensorflow (param class_weight)
* On peut aussi faire du ``sample_weight`` 
    * Chaque échantillon de ``y_train`` recoit une pondération spécifique à sa classe 
    * Voir param ``sample_weight`` de ``model.fit()`` de tensorflow
* Quand les classes sont déséquilibrées, faire attention aux metrics 
    * conf matrix, precision, recall,F1 score, area under ROC curve
* Lire cet [article](https://www.analyticsvidhya.com/blog/2020/07/10-techniques-to-deal-with-class-imbalance-in-machine-learning/)







<!-- 
############################################################
## 
############################################################ 
-->
Question : No category yet - MSE, RMSE, MAE, R², MAPE ?
Answer  :

**RMSE :** 

* Généralement préférée dans les pb de régression
* Si outliers, faut peut-être prendre en compte la MAE
* RMSE (L2 based) est plus sensible aux outliers que MAE (L1 based)
* MAE et RMSE sont 2 mesures de distance entre vecteurs (prédictions et target)
* Différentes mesures de distances sont possibles:
    * RMSE : norme euclidienne, L2
    * MAE : norme de Manhattan, L1
    * Plus le n de Ln augmente et plus la norme focalise sur les grandes valeurs en négligeant les petites
    * Quand les outliers sont exponentiellement rares (bell curve) RMSE est plus efficace

**MAPE :** 

* Erreur Absolue Moyenne en % de la vraie valeur
* Exprimée en %, c'est une mesure simple et intuitive de l'accuracy d'un modèle
* Important de l'utiliser avec prudence quand les valeurs vraies sont proches de zéro.
* MAPE=0% : modèle parfait. 
* MAPE élevé : erreurs importantes dans les prédictions


<!-- 
############################################################
## 
############################################################ 
-->
Question : No category yet - ANOVA... Ca te dis quoi?
Answer  : 

Analyse de la variance. La variation c'est l'information. ANOVA c'est analyser la quantité d'information captée par le modèle. Variance=écart à la moyenne.


<!-- 
############################################################
## 
############################################################ 
-->
Question : No category yet - Silhouette, un commentaire? 
Answer  : 

* Le coefficient de silhouette évalue la cohésion et la séparation des clusters. 
* Paramètre global du clustering. 
* On veut des clusters bien regroupés autour de leur centroïd et bien séparés entre eux. 
* Coef sans unité. 
* Entre -1 et 1. 
* On veut 1 mais 0.5 c'est OK. 
* On choisit k tel que s soit maximal (voir aussi analyse courbe WCSS, Elbow). 
* Le score de silhouette est calculé pour chaque point de données en mesurant la similarité avec son propre cluster par rapport aux clusters voisins les plus proches.

<!-- 
############################################################
## 
############################################################ 
-->
Question : No category yet - Data Warehouse vs Databases ?
Answer  : 

* **Data warehouses** : optimized to have a performance boost on columns (features)
* **Databases**       : optimized for extracting rows (observations)


<!-- 
############################################################
## 
############################################################ 
-->
Question : No category yet - What is data tidying?
Answer  : 

This is the process of transforming raw, messy data into a clean and organized format that is easier to analyze and interpret. This process involves structuring the data in a way that :

* Each variable forms a column
* Each observation forms a row
* Each type of observational unit forms a table



<!-- 
############################################################
## 
############################################################ 
-->
Question : No category yet - What are the types of categorical data ?
Answer  : 

**Ordinal Data** & **Nominal Data**

* Ordinal data is a ranking list. It’s ordered, **BUT** the intervals between the ranks aren’t necessarily equal. 
* Nominal data is like choosing your favorite ice cream flavor. There’s no logical order to the choices. Whether it’s “Vanilla,” “Chocolate,” or “Strawberry,” one isn’t inherently better or worse than the others. 


<!-- 
############################################################
## 
############################################################ 
-->
Question : No category yet - D'où provient le terme recall ?
Answer  : 

* Le terme "recall" en machine learning vient du domaine de la récupération d'information (Information Retrieval). 
* Dans ce contexte, "recall" se réfère à la capacité d'un système à retrouver toutes les occurrences pertinentes dans un ensemble de données. 
* En d'autres termes, le "recall" mesure le pourcentage de vrais positifs parmi tous les éléments pertinents. Il est utilisé pour évaluer la performance des modèles de classification, en particulier dans les situations où il est crucial de ne pas manquer des éléments pertinents (par exemple, dans la détection de maladies, où il est important de ne pas manquer de cas positifs). 


<!-- 
############################################################
## 
############################################################ 
-->
Question : No category yet - Boostrapping ?  
Answer  : 

* On crée plusieurs sous-ensembles de données en échantillonnant de manière aléatoire avec remplacement à partir de l'ensemble de données d'origine. 
* Chaque sous-ensemble peut donc contenir des exemples répétés et ne pas inclure certains exemples de l'ensemble d'origine.


<!-- 
############################################################
## 
############################################################ 
-->
Question : No category yet - Bagging ?
Answer  : 

* Parallèle (indépendance)
* Le bagging consiste à entraîner plusieurs modèles indépendamment les uns des autres sur différentes versions d'un même ensemble de données, puis à combiner leurs prédictions pour obtenir un modèle final plus robuste. 
* Objectif : Réduction de la variance. Exemple : Random Forest
* 3 phases : 
    1. Bootstrap Sampling
    1. Entraînement des modèles
    1. Agrégation des résultats (Classification => agrégation par vote majoritaire. Régression => agrégation par la moyenne des prédictions)



<!-- 
############################################################
## 
############################################################ 
-->
Question : No category yet - Boosting ?
Answer  : 

* Séquentiel
* Le boosting consiste à entraîner plusieurs modèles de manière séquentielle, chaque modèle cherchant à corriger les erreurs des modèles précédents. 
* Les modèles sont construits de manière dépendante et on pondère les exemples d'entraînement en fonction des erreurs des modèles précédents. 
* Objectif : Réduction du biais et de la variance (correction des erreurs progressives). Exemple : XGBoost, AdaBoost
* 3 phases : 
    1. Initialisation          : Un premier modèle est entraîné sur l'ensemble de données d'origine.
    1. Ajustement              : Les exemples mal classés ou mal prédits par le modèle précédent sont pondérés davantage, de sorte que le modèle suivant se concentre sur ces erreurs.
    1. Combinaison des modèles : Les modèles sont combinés en pondérant leurs prédictions en fonction de leur performance. En général, les modèles performants reçoivent un poids plus important.



<!-- 
############################################################
## 
############################################################ 
-->
Question : No category yet - XGBoost ?
Answer  :

* Extreme Grandient Boosting
* Based on decision trees
* Several weak models (decision trees) are combined to form a robust model
* Model 1 predictions are compared with true values
* Each model is trained to minimize a loss function that measures the residual error
* Residuals are kept and become the target values of the next model
* The new set of weighted observations is injected into model 2
* At the end we have n models
* Submit an observation.
* Sum of the n predictions (each tree predict a residual)


<!-- 
############################################################
## 
############################################################ 
-->
Question : No category yet - XGBoost benefits drawbacks for predictions
Answer  :

#### Benefits

* **Handles complex and heterogeneous data**: XGBoost works well with diverse features (numerical, categorical) and can model complex non-linear relationships often found in price prediction tasks.
* **High performance with large datasets**: It is designed to be highly efficient, even with large volumes of data, thanks to parallelization and memory optimization.
* **Manages missing values and outliers**: XGBoost automatically handles missing data and can incorporate outliers without negatively impacting performance, which is crucial for price prediction scenarios.
* **Prevents overfitting**: With regularization techniques, depth control, and learning rate adjustments, XGBoost helps reduce overfitting, making it suitable for price models with high complexity.
* **Feature importance and interpretability**: XGBoost provides insights into feature importance, helping identify key factors that influence price predictions, which is valuable for decision-making.
* **Robust and adaptable**: It performs well across different data distributions and can handle cases where price variance is high, adapting to various relationships between features and target variables.


#### Potential Drawbacks

* **Complex hyperparameter tuning**: To achieve optimal performance, XGBoost often requires careful hyperparameter tuning, which can increase development complexity.
* **Training time**: While fast, training can be longer on very large datasets compared to simpler algorithms, though the test performance typically makes up for this. 



<!-- 
############################################################
## 
############################################################ 
-->
Question : No category yet - Pouvez-vous expliquer les termes "précision" et "accuracy"
Answer  : 

Métriques utilisées pour évaluer la performance des modèles de classification.

* L'accuracy (exactitude) est une mesure globale de la performance d'un modèle. C'est le pourcentage de prédictions correctes sur le total des prédictions : (TP + TN)/(total de prédictions)
* Précision : une mesure la qualité des prédictions positives (précision, positive). C'est le pourcentage de prédictions positives correctes par rapport au nombre total de prédictions positives. TP/(TP+FP). La précision est cruciale lorsque les faux positifs ont un coût élevé, comme dans le dépistage des maladies, où un faux positif pourrait entraîner des tests supplémentaires inutiles.
* L'accuracy donne une idée globale de la performance du modèle, mais peut être trompeuse si les classes sont déséquilibrées. 


<!-- 
############################################################
## 
############################################################ 
-->
Question : No category yet - C'est quoi le F1 score, que signifie-t-il, quand l'utiliser?
Answer  : 

* Moyenne harmonique du recall et de la précision. 
* Utile quand : 
    1. Les classes sont déséquilibrées 
    1. Si les faux positifs et les faux négatifs ont des coûts comparables, le F1 score fournit un bon compromis. 
* Si proche de 1 =>  le modèle a une bonne performance, équilibrant  précision et rappel. 
* Si faible => soit le modèle a une faible précision, soit un faible rappel, soit les deux. Le modèle ne performe pas bien et nécessite des ajustements.


<!-- 
############################################################
## 
############################################################ 
-->
Question : No category yet - Le F1 score est une moyenne harmonique. Pourquoi ?
Answer  : 

* Penser à la moyenne "harmonieuse"
* Elle est maximale quand les valeurs sont identiques
* F1 : on cherche le compromis Recall & Precision 

<p align="center">
<img src="../static/md/assets/harmonic.png" alt="harmonic" width="577"/>
</p>


<!-- 
############################################################
## 
############################################################ 
-->
Question : No category yet - Forward selection, Backward selection ?
Answer  : 

* Forward selection  : On ajoute les variables au modèle qui à chaque étape augmente le R². On arrête si y a plus variable ou si R² baisse.
* Backward selection : Elimination. On part avec toutes les variables. On élimine la variable qui a la plus forte probabilité de ne pas être pertinante (p-value). On arrête quand toutes les variables ont une p-value sup à 5%


<!-- 
############################################################
## 
############################################################ 
-->
Question : No category yet - What is a kernel? 
Answer  : Function that take the observations into a larger dimensional space, in which we hope that the geometric properties of the observations will be linearly separable


<!-- 
############################################################
## 
############################################################ 
-->
Question : No category yet - Objectif de K-Means ?
Answer  : 

* L'objectif du K-Means est de regrouper des données en K clusters de telle manière que les points à l'intérieur d'un cluster soient similaires entre eux et différents des points d'autres clusters
* On fait ça en minimisant la variance intra-cluster. 
* La variance intra-cluster est une mesure de la dispersion des données à l'intérieur de chaque cluster. Elle représente la somme des carrés des distances entre chaque point de données d'un cluster et le centre de ce cluster.


<!-- 
############################################################
## 
############################################################ 
-->
Question : No category yet - WCSS… Any comment ? 
Answer  : 

* Within Cluster Squared Sum. 
* Voir méthode ELBOW (Densité des clusters). 
* Pour chaque exécution de K-Means, on calcule la WCSS (within cluster squared sum, somme des distances au carré entre chaque point de données et le centroïde de son cluster correspondant). 
* C'est un paramètre global sur l'ensemble des clusters. C'est la somme des sommes des carré


<!-- 
############################################################
## 
############################################################ 
-->
Question : No category yet - homoscédasticité ? 
Answer  : 

On parle d'homoscédasticité lorsque la variance des erreurs stochastiques de la régression est la même pour chaque observation i (de 1 à n observations). 


<!-- 
############################################################
## 
############################################################ 
-->
Question : No category yet - Quelles sont les 3 hypothèses que l'on fait en régression linéaire
Answer : 

* Linéarité : évident
* Indépendance des erreurs : l'erreur sur une observation est indépendante de l'erreur sur une autre. Difficile à prouver à partir d'échantillons. Corrélation vs Causation
* Homoscédasticité : La distribution des erreurs est indépendante de y. Faut que la distribution des erreurs, que l'écart à la droite, soit constant qqsoit y


<!-- 
############################################################
## 
############################################################ 
-->
Question : No category yet - What is Boto3 the SDK (Software Development Kit) of AWS?
Answer  : 

It is a collection of tools and libraries to help you use AWS using code


<!-- 
############################################################
## 
############################################################ 
-->
Question : No category yet - What is a RDBMS (Relational DataBase Management System)?
Answer  : 

A piece of software that lets define, create, maintain, and control access to a database


<!-- 
############################################################
## 
############################################################ 
-->
Question : No category yet - C'est quoi le machine learning ? 
Answer  : 

* Machine Learning (ML) = sous-domaine de l'IA 
* Se concentre sur le développement de techniques permettant aux ordinateurs d'apprendre à partir de données et d'améliorer leurs performances sans être explicitement programmés pour chaque tâche. 
* Le ML permet aux systèmes informatiques :  
	1. de reconnaître des modèles dans les données 
	1. de faire des prédictions ou de prendre des décisions basées sur ces modèles
	1. sans intervention humaine directe pour spécifier explicitement les règles.


<!-- 
############################################################
## 
############################################################ 
-->
Question : No category yet - Expliquez ce qu'est la validation croisée et pourquoi c'est important dans le contexte de l'apprentissage automatique ?
Answer  : 

* Technique utilisée pour évaluer les performances d'un modèle en divisant les données en sous-ensembles d'apprentissage et de test de manière itérative. 
* Cela permet d'estimer la capacité de généralisation du modèle sur des données non vues, et d'identifier le surapprentissage. 
* Les méthodes courantes incluent la validation croisée en **k-fold** et la validation croisée **leave-one-out**.





<!-- 
############################################################
## 
############################################################ 
-->
Question : No category yet - Différence entre régression et classification en apprentissage automatique ?
Answer  : 

* La **régression** est utilisée pour prédire une **valeur continue**
* La **classification** est utilisée pour prédire une classe ou une catégorie discrète.








<!-- 
############################################################
## 
############################################################ 
-->
Question : No category yet - Pouvez-vous expliquer ce qu'est l'overfitting et comment le détecter ?
Answer  : 

* L'overfitting se produit lorsque le modèle s'adapte trop précisément aux données d'entraînement et perd sa capacité de généralisation sur de nouvelles données. 
* Il peut être détecté en observant une performance élevée sur les données d'entraînement mais une performance médiocre sur les données de test
* On peut aussi comparer les performances du modèle sur les données d'entraînement et de validation.






<!-- 
############################################################
## 
############################################################ 
-->
Question : No category yet - Quelle est la différence entre la **normalisation** et la **standardisation** des données ?
Answer  : 

* La **normalisation**   : met à l'échelle les données dans une plage spécifique, souvent entre 0 et 1. 
* La **standardisation** : transforme les données pour qu'elles aient une moyenne nulle et un écart-type de 1. Penser à la courbe de gauss.





<!-- 
############################################################
## 
############################################################ 
-->
Question : No category yet - Qu'est-ce qu'une fonction de coût (ou de perte) et comment est-elle utilisée dans l'apprentissage automatique ?
Answer  : 

* Une fonction de coût mesure l'erreur entre les prédictions d'un modèle et les valeurs réelles de l'ensemble de données. 
* Elle est utilisée dans le processus d'optimisation pour guider l'ajustement des paramètres du modèle afin de minimiser cette erreur. 
* Parler de régression => MSE Classification => LogLoss (entropy)





<!-- 
############################################################
## 
############################################################ 
-->
Question : No category yet - Pouvez-vous expliquer ce qu'est la réduction de la dimensionnalité et pourquoi est-ce important dans l'analyse de données ?
Answer  : 

* La **PCA** (princiapl component analysis, analyse en composantes principales) consiste à réduire le nombre de variables ou de caractéristiques dans un ensemble de données. 
* Cela permet de simplifier les modèles, de réduire le temps de calcul et de prévenir le surapprentissage, tout en préservant autant que possible les informations importantes.





<!-- 
############################################################
## 
############################################################ 
-->
Question : No category yet - Quelles sont les différences entre l'apprentissage supervisé et l'apprentissage non supervisé ?
Answer  : 

* L'apprentissage supervisé implique l'utilisation de données étiquetées pour entraîner un modèle à prédire une sortie
* tandis que l'apprentissage non supervisé explore les données pour découvrir des structures intrinsèques sans étiquettes
* Parler des cas d'usage du non supervisé. Pas une fin en soi









<!-- 
############################################################
## 
############################################################ 
-->
Question : No category yet - Expliquez ce qu'est la régularisation **Lasso** et en quoi elle diffère de la régularisation **Ridge** ?
Answer  : 

* **Lasso :** La régularisation Lasso ajoute, à la fonction de coût, une pénalité proportionnelle à la *valeur absolue des coefficients du modèle*, ce qui favorise la sélection de caractéristiques importantes et conduit à une certaine sparsité. 
* **Ridge :** La régularisation Ridge utilise une pénalité proportionnelle au *carré des coefficients*, ce qui réduit la magnitude des coefficients sans les éliminer complètement.




<!-- 
############################################################
## 
############################################################ 
-->
Question : No category yet - C'est quoi la régularisation ?
Answer  : 

* Une technique pour réduire le surapprentissage (overfitting) en pénalisant les modèles trop complexes. 
* Consiste à ajouter un terme de pénalité à la fonction de coût lors de l'entraînement. 
* Ca encourage le modèle à privilégier des solutions plus simples. 
* La régularisation aide à améliorer la généralisation du modèle en contrôlant sa complexité et en réduisant le risque de surapprentissage.









<!-- 
############################################################
## 
############################################################ 
-->
Question : No category yet - Elbow method, any comment?
Answer  : 

Dans le contexte de l'algorithme K-Means (non supervisé), la méthode Elbow est utilisée pour déterminer le nombre optimal de clusters à utiliser. 

1. Exécution de K-Means pour différents nombres de clusters (de 2 à 20 par exemple). 
2. Calcul de la variance intra-cluster (inertie)
    * Pour chaque exécution de K-Means, on calcule la WCSS 
    * WCSS = within cluster squared sum, somme des distances au carré entre chaque point de données et le centroïde de son cluster correspondant. 
    * Mesure de la dispersion des données à l'intérieur de chaque cluster. 
    * L'inertie intra-cluster. 
    * Plus le nombre de clusters est élevé, plus l'inertie intra-cluster tend à diminuer (si y a autant de cluster que de points elle vaut 0)
    * WCSS est un para global à l'ensemble des clusters. C'est la somme des sommes des carrés
3. Tracé du graphique inertie intra-cluster vs nombre de clusters. 
4. Identification du point de coude sur le graphe. 
    * On recherche le point où la décroissance de l'inertie intra-cluster commence à ralentir de manière significative. 
    * C'est le point où ajouter un cluster de plus k=k+1 ne fait pas basser WCSS de manière significative. 
    * Cela ressemble à un coude sur le graphique. 
    * Ce point est souvent considéré comme le nombre optimal de clusters à utiliser. 
    * Dans certains cas, le point de coude peut ne pas être clairement défini. 
        * Il peut être utile alors d'utiliser d'autres méthodes de validation des clusters (Silhouette).


<!-- 
############################################################
## 
############################################################ 
-->
Question : No category yet - Compromis biais-variance... Ca te parle ?
Answer  : 

1. Le biais mesure à quel point les prédictions d'un modèle diffèrent des valeurs réelles. 
    * Un modèle avec un biais élevé simplifie trop les données d'entraînement et sous-estime la complexité de la relation entre les features et la target. 
    * Conduit à des performances médiocres sur les données d'entraînement et de test. 
    * Les modèles à haut biais sont généralement trop simples pour capturer la complexité des données. 
    * Pour **réduire le biais**, on peut 
        * utiliser des modèles plus complexes 
        * augmenter la taille 
        * augmenter la complexité des caractéristiques utilisées. Features engineering.
2. La variance mesure la sensibilité d'un modèle aux petites variations dans l'ensemble de données d'entraînement. 
    * Un modèle avec une variance élevée est trop sensible au bruit dans les données d'entraînement
    * Cela peut conduire à un surajustement. 
    * Le modèle fonctionne bien sur les données d'entraînement mais il a du mal à généraliser sur de nouvelles données. 
    * Les modèles à haute variance sont souvent complexes (arbres de décision profonds, réseaux neuronaux avec de nombreux paramètres). 
    * Pour **réduire la variance**, on peut utiliser 
        * la régularisation
        * la réduction de la dimensionnalité 
        * augmentation des données.

La **validation croisée** peut être utile pour évaluer comment le compromis biais-variance affecte les performances du modèle. En utilisant la validation croisée, on peut ajuster les hyperparamètres du modèle pour trouver le meilleur compromis entre biais et variance.


<!-- 
############################################################
## 
############################################################ 
-->
Question : No category yet - Connaissez-vous la différence en Corrélation et Causalité ? Pouvez-vous citer quelques critères à vérifier ?
Answer  : 

1. Temporalité : La cause doit précéder l'effet dans le temps.
1. Force de l'association : Une forte association entre une cause potentielle et un effet observé renforce l'idée que la relation pourrait être causale plutôt que simplement corrélée.
1. Plausibilité biologique : Il doit exister une explication scientifique ou un mécanisme pour relier les variables.
1. Relation dose-réponse : Une augmentation de l'exposition doit conduire à une augmentation de l'effet (si la relation est causale).
1. Expérimentation : Des expériences contrôlées, comme des essais randomisés, permettent de tester la causalité.
1. Réversibilité : Si la cause est supprimée, l'effet devrait également disparaître ou diminuer.
1. Consistance : Les résultats doivent être reproduits dans différentes études ou contextes.
1. Spécificité : Un effet spécifique doit être attribué à une cause spécifique.


<!-- 
############################################################
## 
############################################################ 
-->
Question : No category yet - Dans le cas du Machine Learning et du Deep Learning êtes vous d'accord pour dire qu'on laisse le modèle trouver des corrélations et pas des relations de cause à effet?
Answer  : 

**Réponse courte :** Oui car un réseau de neurones peut prédire avec qu'il pleuvra demain si les nuages sont présents, mais cela ne signifie pas qu'il comprend que les nuages **causent** la pluie.

Quoiqu'il en soit, oui oui, je suis d'accord. En ML/DL, les modèles se concentrent sur l'identification de **corrélations** plutôt que sur l'établissement de **relations causales**.

1. **Objectif des modèles ML/DL** : 
   Les algorithmes d'apprentissage supervisé, non supervisé ou par renforcement apprennent à partir des données pour trouver des motifs, des régularités et des corrélations entre les variables d'entrée (features) et les sorties (labels ou classes). Ces corrélations permettent au modèle de faire des prédictions, par exemple, dans la reconnaissance d'images ou la classification de textes. Ces algorithmes ne sont pas conçus pour établir un lien de cause à effet entre les variables.

2. **Corrélations vs Causalité** :
   - **Corrélation** : Les modèles ML/DL détectent des relations statistiques. Ils peuvent trouver des associations entre les données, même si ces associations sont dues au hasard ou à des variables non observées.
   - **Causalité** : Déterminer la causalité nécessite non seulement d'identifier des relations statistiques, mais aussi d'établir un mécanisme expliquant **comment** et **pourquoi** une variable A affecte une variable B. Cela demande généralement une intervention expérimentale (comme les essais contrôlés randomisés), ou des techniques statistiques spécifiques aux modèles causaux.

3. **Pourquoi le DL ne traite pas la causalité directement** :
   - Les réseaux neuronaux (qui sont au cœur du *deep learning*) sont souvent des boîtes noires, c'est-à-dire qu'il est difficile de comprendre **comment** ils arrivent à leurs conclusions. Ils ne sont pas conçus pour interpréter ou expliquer les relations causales. Leur objectif est plutôt de minimiser les erreurs de prédiction à partir des corrélations dans les données d'entraînement.
   - Par exemple, un réseau de neurones peut prédire avec précision qu'il pleuvra demain si les nuages sont présents, mais cela ne signifie pas qu'il comprend que les nuages **causent** la pluie.

4. **Modèles de causalité en ML** :
   Bien que les modèles standards ML/DL ne cherchent pas directement à établir la causalité, il existe des sous-domaines spécialisés du machine learning, comme le **causal inference** ou les **modèles causaux** (e.g., les graphiques de causalité basés sur les travaux de Judea Pearl), qui visent à analyser et à comprendre les relations causales en plus des corrélations.


<!-- 
############################################################
## 
############################################################ 
-->
Question : No category yet - Etes vous d'accord (ou pas) pour dire qu'une IA peut conseiller (il y a x% de chance que l'image soit celle d'un chat) mais qu'elle ne peut pas expliquer (l'image est un chat car...)

Answer  : 
Oui, oui. Un modèle (ML/DL) peut donner des recommandations ou des prédictions basées sur des corrélations, comme « Il y a 85 % de chances que cette image soit celle d'un chat ». Elle se base sur les motifs dans les données qu’elle a appris pendant l'entraînement.

Cependant, elle ne peut pas expliquer le raisonnement derrière cette prédiction de manière causale. Elle ne dit pas « C'est un chat parce qu'il a des oreilles pointues, des moustaches, etc. ». Même si elle détecte ces caractéristiques, elle ne comprend pas **pourquoi** ces éléments définissent un chat. C'est dû à la nature des algorithmes de deep learning, qui se contentent d'optimiser les prédictions sans modéliser explicitement les liens de cause à effet.

C’est un défi de l’IA. Voir les recherches sur l’IA explicable (XAI, Explainable AI).


<!-- 
############################################################
## 
############################################################ 
-->
Question : No category yet - Entre moyenne et médiane. Laquelle des deux est le plus impacté par les outliers? Peux tu illustrer avec un exemple ?

Answer   : 

* La moyenne est plus impactée par les outliers que la médiane.
* Si je suis au bar, que Bill Gates arrive et qu'on calcule la moyenne des salaires... "On sent bien" que cette dernière va être relativement haute. Et ce même si un de mes collègues nous rejoint. 
* Si on calcule alors la médiane de nos 3 salaires, "on sent bien" que cette dernière va se rapprocher de mon collègue et moi.



<!-- 
############################################################
## 
############################################################ 
-->
Question : No category yet - Comment gères-tu le problème de surapprentissage (overfitting) dans un modèle de machine learning ?

Answer   : 

* Régularisation (L1/L2) pour réduire l'effet de la complexité du modèle
* Validation croisée (cross-validation) pour s'assurer que le modèle généralise bien
* Pruning (réduction de la taille du modèle)
* Utiliser des techniques comme dropout dans les réseaux de neurones
* Collecter plus de données ou réduire le nombre de variables d'entrée (feature selection, PCA)


<!-- 
############################################################
## 
############################################################ 
-->
Question : No category yet - Comment choisir entre une régression linéaire et un modèle plus complexe comme les forêts aléatoires (Random Forest) ?

Answer   : 

Ca va dépendre de la nature des données
* **Régression linéaire :** utile si la relation entre les variables d'entrée et de sortie est linéaire ou quasi-linéaire. Modèle simple, facile à interpréter et rapide à entraîner.
* **Forêt aléatoire :** utile lorsque les relations sont plus complexes et non linéaires. Moins sensible aux variables bruitées et outliers. Plus difficile à interpréter. Préférable quand la performance prime sur l'interprétabilité.

Ne pas oublier de mentioner que Scikit-Learn est construit de telle sorte qu'il est très facile de mettre en oeuvre plusieurs modèles avec le même code. Autrement dit, il ne faut pas hésiter à faire des tests et à comparer les résultats des différents modèles. Il faudra alors faire un choix sur LA métrique à laquelle il faudra accorder le plus d'importance.





<!-- 
############################################################
## 
############################################################ 
-->
Question : No category yet - Quelles sont les étapes typiques d'un projet de machine learning ?

Answer   : 

1. Collecte des données : 
1. Prétraitement des données : Nettoyage, gestion des valeurs manquantes, et transformation des données (normalisation, encodage des catégories).
1. Séparation des données : Séparer les données en ensembles d'entraînement, de validation et de test.
1. Choix du modèle : Sélectionner un ou plusieurs algorithmes de machine learning adapté au problème.
1. Entraînement du modèle : 
1. Évaluation du modèle : Utiliser les données de validation/test pour évaluer sa performance (précision, rappel, AUC, etc.).
1. Optimisation : Ajuster les hyperparamètres, utiliser des techniques comme la régularisation pour améliorer les résultats.
1. Déploiement : Mettre le modèle en production
1. Monitoring du modèle : Métriques du modèle + drift dans les données (Evidently AI)
1. CI/CD pour le machine learning : Penser à automatisation le roll-back en cas d’échec d’un déploiement

**Philosophie :**

* Mettre en oeuvre toute la chaîne autour d'un BaseLine Model rapidement. Il sera toujours temps de faire évoluer le modèle ensuite.
* Pour le modèle. Ne pas réinventer la roue. Voir ce qui a été fait, ce qui marche et comment sont obtenus les meilleurs résultats. Transfer Learning



<!-- 
############################################################
## 
############################################################ 
-->
Question : No category yet - Comment gères-tu les biais dans les données et comment t’assures-tu de l’équité dans les modèles ?

Answer   : 

1. Analyse des biais dans les données en amont : identifier les déséquilibres dans les classes, les caractéristiques démographiques, ou autres facteurs.
1. Utilisation de techniques comme la re-sampling (over-sampling/under-sampling) ou la pondération pour traiter les déséquilibres.
1. Audits de modèles réguliers pour surveiller les performances du modèle sur des sous-groupes sensibles.
1. L'application de métriques d'équité comme l'indice de parité de traitement ou la parité démographique.
1. Des approches pour corriger ces biais après le déploiement, telles que algorithmes de post-processing pour ajuster les décisions du modèle.


<!-- 
############################################################
## 
############################################################ 
-->
Question : No category yet - Quelle est ton approche pour mettre à l'échelle un modèle de machine learning en production ?

Answer   : 

* Optimisation des performances : réduire la latence du modèle via la quantification des modèles (réduire la précision des poids), ou distillation de modèle pour créer des versions plus légères.
* Utilisation de frameworks distribués comme Spark MLlib ou TensorFlow Distributed pour gérer des volumes de données importants.
* Déploiement sur des environnements distribués ou dans le cloud (Kubernetes, AWS SageMaker, Google AI Platform) et utilisation de systèmes de streaming pour traiter des données en temps réel.
* Stratégies de mise en cache, de gestion de clusters, et d'utilisation de pipelines CI/CD pour le déploiement continu.
* Méthodologies pour surveiller et ajuster les modèles après leur mise en production.