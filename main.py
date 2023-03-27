from RegressionLogistique import *
from UsualFunctions import *
import joblib
from Caracteristiques import *
from SVM import *
from KNN import *

"""
Created on Fri Jan 20 19:07:43 2023
Groupe Sierra : Nesrine GHANNAY, Simon HIATY, Gael TUCZAPSKI et Bryce MANTILARO
@author: cecile capponi
"""

# Permet de prédire les labels des images contenus dans le dossier "TestCC2" avec le model appris "LogisticRegression.pkl".
# Le fichier produit sera nommé "PredictionCC2_LG.txt.txt"
# classifyingImages('TestCC2', 'LogisticRegression.pkl', "PredictionCC2_LG.txt")



# Permet de prédire les labels des iamges dans le dossier "TestCC2 avec le model appris "SVM.pkl"
# Le fichier produit sera nommé "PredictionCC2_SIERRA.txt"
# mainSVM_prediction("SVM.pkl", "TestCC2", "PreditionCC2_SVM.txt")




# Algorithme non retenu, KNN

#run_knn_classification("C:/Users/bryce/PycharmProjects/projet-iaa/Data","C:/Users/bryce/PycharmProjects/projet-iaa/TestCC2","C:/Users/bryce/PycharmProjects/projet-iaa",'HC')

