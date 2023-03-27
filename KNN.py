from sklearn.neighbors import KNeighborsClassifier
from UsualFunctions import *


"""
@autors : Bryce
Input : Train_data = image_data = [ dicoImage1 = {nom : ... ; label : ... ; représentation : .....} ; dicoImage2 ....]
(Dictionnaire retourné par une des versions de load_transform_label_from_data) et  
Sortie : Modèle entraîné avec les paramètres de k-neighbors (généralement 5 car c'est le paramètre le plus optimisé pour k-NN)
"""
def learn_knn_model_from_data(train_data, n_neighbors):
    X_train = []
    Y_train = []
    for image in train_data:
        X_train.append(image['representation'])
        Y_train.append(image['label'])
    model = KNeighborsClassifier(n_neighbors=n_neighbors,weights='uniform')
    return model.fit(X_train,Y_train)


"""
@autors : Bryce
Input : Train_data, cv (number of cross validation), metric(scoring) = 'uniform',  or 'accuracy', accuracy is better from multiples tests
Output : Evaluation of the model in percent
CAUTION: #Possibilité ici de modifier les paramètres par défaut et de prendre (comme vu dans le rapport) une distance différente de celle
par défaut: cf minkowski, et ici plutôt prendre la distance de manhathan.
"""

def estimate_model_score_cross(train_data, cv, scoring):
    # Préparation des données
    X_train = [d["representation"] for d in train_data]
    Y_train = [d["label"] for d in train_data]

    # Création du modèle kNN
    model = KNeighborsClassifier(n_neighbors=5,weights='uniform')

    # Validation croisée pour estimer la performance du modèle
    scores = cross_val_score(model, X_train, Y_train, cv=cv, scoring=scoring)

    # Calcul de la moyenne des scores pour chaque pli
    return scores.mean()


"""
### First function: Blue data vs normal data
Allows to compare the trained model by varying a pre-processing which consists in taking only the blue intensities in the color histogram.
@autors : Bryce
Input : Data_path, representation = 'HC', k_fold = 5, metric
Output : Evaluation of the model in percent with two variations: with and without blue caracteristic 
"""

def run_cross_validation(data_path, representation, k_fold=5, metric='accuracy'):
    data = load_transform_label_train_data(data_path, representation)
    data_normalized = normalize_representation(data)
    data_blue = extract_blue_channel(data_normalized)
    print("Validation croisée, toutes les données: ", estimate_model_score_cross(data_normalized, k_fold, metric))
    print("Validation croisée, données bleues seulement: ",  estimate_model_score_cross(data_blue, k_fold, metric))

"""
### Second function: Blue data vs all data + model trained with augmented data
### We should not use data_4 which falsifies the accuracy of the model since the intensity of the pixels does not change if we rotate the image
(sur-entrainement)
@autors : Bryce
Input : data_path, representation, k-fold, metric
Output :  Evaluation of the model in percent with two variations: with and without blue caracteristic  + augmented data (random crop, zoom and rotations)
"""

def run_cross_validation_2(data_path, representation, k_fold=5, metric='accuracy'):
    data_1 = load_transform_label_train_data(data_path, representation)
    data_2 = load_transform_label_train_data_zoom(data_path, representation)
    data_3 = load_transform_label_train_data_crop(data_path, representation)
    #data_4 = load_transform_label_train_data_rotations(data_path, representation)

    data = data_1 + data_2 + data_3
    data_normalized = normalize_representation(data)
    data_blue = extract_blue_channel(data_normalized)

    print("Validation croisée, toutes les données: ", estimate_model_score_cross(data_normalized, k_fold, metric))
    print("Validation croisée, données bleues seulement: ", estimate_model_score_cross(data_blue, k_fold, metric))

"""
### Third function: Training of the KNN model with the chosen parameters + test and normal data separation + write predictions
@autors : Bryce
Input :data_path, test_path,path_for_predictions,representation, k_neighbors=5
Output : Creation of the predictions.txt file with the data predicted by predicted_labels of the model, + cross_val to get a score from the model.
"""

def run_knn_classification(data_path, test_path,path_for_predictions,representation, k_neighbors=5):
    data_1 = load_transform_label_train_data(data_path, representation)
    data_2 = load_transform_label_train_data_zoom(data_path, representation)
    data_3 = load_transform_label_train_data_crop(data_path, representation)
    #data_4 = load_transform_label_train_data_rotations(data_path, representation)

    data = data_1 + data_2 + data_3
    data_normalized = normalize_representation(data)

    test_data_1 = load_transform_test_data(test_path, representation)
    test_data = test_data_1
    test_data_normalized = normalize_representation_dict(test_data)

    model = learn_knn_model_from_data(data_normalized, k_neighbors)

    algo_dico = {'algorithm': 'knn', 'n_neighbors': k_neighbors}
    k = 5 # nombre de plis pour la validation croisée
    # print("Modèle entrainé, validation croisée:" , estimate_model_score(model,data_normalized,k))

    predicted_labels = predict_sample_label_2(test_data_normalized, model)
    write_predictions(path_for_predictions, predicted_labels,'PredictionsCC2_kNN.txt')

# run_knn_classification("Data", "TestCC2", "Predictions", "HC")