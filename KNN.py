from sklearn.neighbors import KNeighborsClassifier
from UsualFunctions import *

def learn_knn_model_from_data(train_data, n_neighbors):
    X_train = []
    Y_train = []
    for image in train_data:
        X_train.append(image['representation'])
        Y_train.append(image['label'])
    model = KNeighborsClassifier(n_neighbors=n_neighbors,weights='uniform')
    return model.fit(X_train,Y_train)



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

### Première fonction: Données bleues vs données normales

def run_cross_validation(data_path, representation, k_fold=5, metric='accuracy'):
    data = load_transform_label_train_data(data_path, representation)
    data_normalized = normalize_representation(data)
    data_blue = extract_blue_channel(data_normalized)
    print("Validation croisée, toutes les données: ", estimate_model_score_cross(data_normalized, k_fold, metric))
    print("Validation croisée, données bleues seulement: ",  estimate_model_score_cross(data_blue, k_fold, metric))


### Deuxième fonction:  Données bleues vs toutes les données + modèle entrainé avec augmented data

def run_cross_validation_2(data_path, representation, k_fold=5, metric='accuracy'):
    data_1 = load_transform_label_train_data(data_path, representation)
    data_2 = load_transform_label_train_data_zoom(data_path, representation)
    data_3 = load_transform_label_train_data_crop(data_path, representation)
    data_4 = load_transform_label_train_data_rotations(data_path, representation)

    data = data_1 + data_2 + data_3 + data_4
    data_normalized = normalize_representation(data)
    data_blue = extract_blue_channel(data_normalized)

    print("Validation croisée, toutes les données: ", estimate_model_score_cross(data_normalized, k_fold, metric))
    print("Validation croisée, données bleues seulement: ", estimate_model_score_cross(data_blue, k_fold, metric))


### Troisième fonction : Aprentissage du modèle KNN avec les paramètres choisis + séparation test et data normal + write predictions


def run_knn_classification(data_path, test_path, representation, k_neighbors=5):
    data_1 = load_transform_label_train_data(data_path, representation)
    data_2 = load_transform_label_train_data_zoom(data_path, representation)
    data_3 = load_transform_label_train_data_crop(data_path, representation)
    data_4 = load_transform_label_train_data_rotations(data_path, representation)

    data = data_1 + data_2 + data_3 + data_4
    data_normalized = normalize_representation(data)

    test_data_1 = load_transform_test_data(test_path, representation)
    test_data = test_data_1
    test_data_normalized = normalize_representation_dict(test_data)

    model = learn_knn_model_from_data(data_normalized, k_neighbors)

    algo_dico = {'algorithm': 'knn', 'n_neighbors': k_neighbors}
    k = 5 # nombre de plis pour la validation croisée
    print("Modèle entrainé, validation croisée:" , estimate_model_score(model,data_normalized,k))

    predicted_labels = predict_sample_label_2(test_data_normalized, model)
    write_predictions(os.path.join(data_path, "Predictions"), predicted_labels,'predictions.txt')
