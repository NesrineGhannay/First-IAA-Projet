# <span style="color: #086F61"> Projet d'introduction à l'apprentissage automatique : Projet de Drone </span>

<span style="color: #85BDB5">Groupe SIERRA : Nesrine, Gael, Bryce, Simon      
L3 Informatique parcours mathématiques-informatique (groupe 3.1)

## <span style="color: #086F61"> Objectifs
L'objectif de ce projet est de construire un classifieur permettant de déterminer si une image représentait une mer, ou non ; tout en minimisant le taux d'erreur. 

## <span style="color: #086F61">Description des différents fichiers 
Le code est fractionné en plusieurs parties : 

* Dans le fichier **UsualFunction.py** se trouve les fonctions usuelles partagées avec les autres fichiers.py qui contiennent les fonctions de bases utilisées par les différents algorithmes d'apprentissage. 


* Dans le fichier **kNN.py** se trouve l'algorithme kNN implémenté que l'on peut lancer en décommentant la dernière ligne de ce même fichier. 


* Dans le fichier **RegressionLogistique.py** se trouve l'algorithme de Régression logistique implémenté. Afin de faire des prédictions avec le model de régression logistique, il suffit de décommenter et de lancer la ligne 16 du main.py pour classifier les image contenus dans le dossier "TestCC2". Pour classifier d'autres images il suffit de mentionner dans quel dossier elles se trouvent à la place de "TestCC2. 


* Dans le fichier **SVM.py** se trouve l'algorithme de Régression logistique implémenté. Afin de faire des prédictions avec le model SVM il suffit de décommenter et de lancer la ligne 16 du main.py pour classifier les image contenus dans le dossier "TestCC2". Pour classifier d'autres images il suffit de mentionner dans quel dossier elles se trouvent. 


* Dans le fichier **Caracteristique.py** se trouve certaine caractéristique, prétraitements implémentés qui ne font pas partis des prétraitements retenus. 


* Dans le dossier **Model** se trouvent les deux models entrainé et sauvegardé SVM.pkl et LogisticRegression.pkl qui ont été construit à partie des données contenue dans le dossier "Data" avec respectivement l'algorithme de Regression Logistique (via RegressionLogistique.py), et d'autre part l'algorithme SVM (contenu dans SVM.py)


* Dans le dossier **Predictions** se trouvent trois fichiers.txt contenant chacun les prédictions faites lors du test de CC2; qui correspond à la prédiction des labels des données contenue dans le fichier "TestCC2.py". 
Les trois fichiers sont obtenus avec le model kNN, LogisticRegression.pkl et enfin SVM.pkl

## <span style="color: #086F61"> Code principal
Etant données que de meilleurs résultats ont été observés sur l'algorthme SVM (détail dans le rapport), ce dernier a été choisi comme étant le model principal du projet. 
Son implémentation se trouve comme évoqué précédemment dans le fichier SVM.py qui utilise certaines fonctions de UsualFunction.py
Le fichier de prédictions faites grâce au model SVM.pkl est enregistré sous le nom de "Sierra.txt" dans le dossier "Predictions". 

## <span style="color: #086F61"> Production finale (comment lancer le code)

### Pour prédire le labels de données contenues dans un dossier
```
Afin de prédire le labels de données contenues dans un dossier, lancez dans le main la commande suivante : 
mainSVM_prediction("SVM.pkl", NOM DU DOSSIER CONTENANT LES IMAGES A PREDIRE, NOM SOUS LEQUEL VOUS VOULEZ ENREGISTRER LES PREDICTIONS)
Sortie : fichier.txt des prédictions affectués qui se trouvera dans le dossier "Predictions"

```

### Pour estimer le score de notre modèle (SVM)
```
Afin d'estimer le score de notre modèle (SVM) en validation croisée lancez dans le main la commande suivante : 
print(estimate_SVM_score("Data"))
Sortie : un pourcentage qui représente le taux d'apprentissage de notre modèle 
```


