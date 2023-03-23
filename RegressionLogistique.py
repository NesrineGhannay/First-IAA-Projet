from UsualFunctions import *
from joblib import *




"""
permet de récupérer les données sous forme de vecteur après les avoir rognée et y avoir extrait la caractéristique blue 
"""
def getFinalData_croped_blue(fileData):
    # transformation et labélisation des données
    data = load_transform_label_train_data_croped(fileData, 'HC')
    # normalisation des données
    data_normalized = normalize_representation(data)
    # Extraction de la caractéristique bleu
    data_blue = extract_blue_channel(data_normalized)
    # mise sous forme de vecteurs des données
    vector_data = transform_to_vecteur(data_blue)
    return vector_data


"""
A partir d'un fichier de données pris en paramètre apprend grâce à la regression logistique un model
"""
def logisticRegression(filedata):
    data = getFinalData_croped_blue(filedata)
    # Création d'une instance du modèle LogisticRegression avec C:  l'inverse de la force de régularisation,
    # liblinear l'algorithme d'optimisation utilisé pour entraîner le modèle et max_iter: le nombre maximum
    # d'itérations autorisées pour la convergence
    lr = LogisticRegression(C=0.8, solver="liblinear", max_iter=100)
    # apprentissage du model
    learnedModel = learn_model_from_data(data, lr)
    return learnedModel

# def getData_augmented(fileData):
#         image_data = load_transform_label_train_data_svm(fileData)
#         return image_data



"""
Permet d'enregistrer le model de regression logistique appris 
"""
def saveLogisticRegression(fileData):
    modelLearned = logisticRegression(fileData)
    # Save the model as a pickle in a file
    saveModel(modelLearned, "LogisticRegression")

# saveLogisticRegression("Data")


'''
Permet d'obetenir le pourcentage de réussite de notre selon le model appris et le fichier sur lequel nous avons appris le model
'''
def score_algo(learnedModel, fileData):
    data = getFinalData_croped_blue(fileData)
    return estimate_model_score(learnedModel, data, 5)


# 0.7094916250367322 (image coupée et où l'on a extrait le paramètre blue)
# Score en augmentant les données : 0.6988859854215377