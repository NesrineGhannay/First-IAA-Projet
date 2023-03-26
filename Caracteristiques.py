import numpy as np
from sklearn.preprocessing import StandardScaler
from UsualFunctions import *


def motie_haute(image):
    width, height = image.size
    half_height = height // 2
    bbox = (0, half_height, width, height)
    img_moitie_basse = image.crop(bbox)
    # img_moitie_basse.show()
    return img_moitie_basse


"""Récupération des donnee images où on a extrait la caractéristique suivante : on prend le bas de l'image
puisqu'en général il est plus problable que la mer s'y trouve"""
def lower_crop_image(image):
    # Size of the image in pixels (size of original image)
    width, height = image.size

    # Setting the points for cropped image
    left = 0
    top = height / 2
    right = width
    bottom = height

    # Cropped image of above dimension
    img = image.crop((left, top, right, bottom))

    return img



def extract_blue_channel(data):
    new_data = []
    for d in data:
        if 'label' in d:
            label = d['label']
        else:
            label = None
        representation = d['representation'][512:768]  # canneaux de bleu
        nom = d['nom']
        new_data.append({'nom': nom, 'representation': representation, 'label': label})
    return new_data


def normalize_representation(image_data):
    scaler = StandardScaler()
    representations = [d["representation"] for d in image_data]
    max_len = max(len(x) for x in representations)
    representations_resized = [np.resize(x, (max_len,)) for x in representations]
    X_normalized = scaler.fit_transform(representations_resized)
    for i, d in enumerate(image_data):
        d["representation"] = X_normalized[i]
    return image_data




"""
@author : Nesrine 
Recovers in a data set train each feature and its associated label
input: the dictionary table corresponding to our training data 
output: a couple X, y; where X is all of our image representation and y is all of the labels associated with each representation
"""
def get_X_y(data_dico):
    X = []
    y = []
    for image in data_dico:
        X.append(image['representation'])
        y.append(image['label'])
    return X, y


"""
@author: Nesrine 
Allows to recover from the dictionary table containing all our processed training data, the characteristic of wave contour via "la transformé de fourrier rapide"
input: the training data dictionary table
output: the same dictionary table where for each representation the wave contour feature has been extracted
"""
def extract_fft(data):
    X_data, y_data = get_X_y(data)
    features = []

    for representation in X_data:
        # appliquer la transformation de Fourier
        fourrier_data = np.fft.fft2(representation)

        # décaler le centre de la transformation
        fshift_data = np.fft.fftshift(fourrier_data)

        # spectre de magnitude
        magnitude_spectrum = 20 * np.log(np.abs(fshift_data))

        features.append(magnitude_spectrum)

    # redimensionner les caractéristiques pour former une matrice d'entraînement
    features = features.reshape(-1, 1)

    return features


# datas_dico = load_transform_label_train_data("Data", 'GC')
# extract_fft(datas_dico)
