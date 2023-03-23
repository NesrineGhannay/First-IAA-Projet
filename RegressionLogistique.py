from UsualFunctions import *


"""
@author: Nesrine
recovers data as a vector after cropping and extracting the blue characteristic
input: the learning data file
output: data transformed after data extraction
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
@author: Nesrine 
From a data file taken as a parameter, learns through logistic regression a model
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



"""
@author: Nesrine
Saves the learned logistic regression model
input: the name of the training file
"""
def saveLogisticRegression(fileData):
    modelLearned = logisticRegression(fileData)
    # Save the model as a pickle in a file
    saveModel(modelLearned, "LogisticRegression")

# saveLogisticRegression("Data")


'''
@author: Nesrine
Allows to keep the success percentage of our model
input: the learned model and the learning file
output: a percentage representing the learning rate of the model taken as a parameter
function used: getFinalData_croped_blue() & estimate_model_score()
'''
def score_algo(learnedModel, fileData):
    data = getFinalData_croped_blue(fileData)
    return estimate_model_score(learnedModel, data, 5)
