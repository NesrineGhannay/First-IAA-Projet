from RegressionLogistique import *
from UsualFunctions import *
import joblib
from Hyperparametres import *




"""
Calcul les prédictions pour les images stockés dans un fichier
input = nom du fichier contenant les images à produire ainsi que l'algorithme d'apprentissage utilisé
output = un fichier .txt contenant toutes les prédictions faites grâce à l'algorithme d'apprentissage choisit
"""
def classifyingImages(fileToClassify, modelFile, nameFilePrediction):
    # Charge le model appris
    model = loadLearnedModel(modelFile)
    # transformation des données test
    testData = load_transform_test_data(fileToClassify, 'HC')
    # normalisation des données
    testDataNormalized = normalize_representation(testData)
    # Extraction de la caractéristique bleu
    blueData = extract_blue_channel(testDataNormalized)
    # mise sous forme de vecteurs des données
    # vectorTestData = transform_to_vecteur(blueData)

    predictData = predict_sample_label(blueData, model)
    # print("PredictedData", predictData)
    return write_predictions("Predictions", predictData, nameFilePrediction)


# Permet de prédire les labels des images contenus dans le dossier "TestCC2" avec le model appris "LogisticRegression.pkl".
# Le fichier produit sera nommé "SIERRA.txt"
# classifyingImages('TestCC2', 'LogisticRegression.pkl', "SIERRA")

def score_SVM(filedata):
    modelSVM = SVM_Algorithm(filedata)
    return estimate_model_score(modelSVM, getFinalData(filedata), 5)

# print(score_SVM("Data"))
