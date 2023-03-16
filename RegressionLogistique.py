from UsualFunctions import *
from joblib import Parallel, delayed
import joblib



def logisticRegression(filedata):
    data = getFinalData(filedata)
    # Création d'une instance du modèle LogisticRegression avec C:  l'inverse de la force de régularisation,
    # liblinear l'algorithme d'optimisation utilisé pour entraîner le modèle  et max_iter: le nombre maximum
    # d'itérations autorisées pour la convergence
    lr = LogisticRegression(C=1.0, solver="liblinear", max_iter=100)
    # apprentissage du model
    learnedModel = learn_model_from_data(data, lr)
    return learnedModel

def saveLogisticRegression(fileData):
    modelLearned = logisticRegression(fileData)
    # Save the model as a pickle in a file
    joblib.dump(modelLearned, 'LogisticRegression.pkl')



'''
Permet d'obetenir le pourcentage de réussite de notre selon le model appris et le fichier sur lequel nous avons appris le model
'''
def score_algo(learnedModel, fileData):
    data = getFinalData(fileData)
    return estimate_model_score(learnedModel, data, 5)



