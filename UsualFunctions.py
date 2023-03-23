# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import json
import pickle

from sklearn.datasets import load_sample_image
from sklearn.model_selection import cross_val_score, learning_curve, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from skimage.transform import rescale, resize, downscale_local_mean
from sklearn.linear_model import LogisticRegression
import joblib
from Caracteristiques import *

from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, make_scorer
import random


"""
Computes a representation of an image from the (gif, png, jpg...) file 
representation can be (to extend) 
'HC': color histogram
'PX': tensor of pixels
'GC': matrix of gray pixels 
other to be defined
input = an image (jpg, png, gif)
output = a new representation of the image
"""
def raw_image_to_representation(image, representation):
    if not image.endswith((".jpg", ".png", ".jpeg", ".jfif", ".JPG")):
        return "Le fichier n'est pas une image valide"

    img = Image.open(image)
    width = 200
    height = 200
    image_redim = img.resize((width, height))

    if representation == 'HC':
        return histo_image(image_redim)

    return "Représentation non disponible"


"""
Computes a representation of an image from the (gif, png, jpg...) file 
representation can be (to extend) 
'HC': color histogram
'PX': tensor of pixels
'GC': matrix of gray pixels 
other to be defined
input = an image (jpg, png, gif)
output = a new representation of the image croped
"""
def raw_croped_image_to_representation(image, representation):
    # Vérification que le fichier est une image (en utilisant une extension d'image telle que .jpg ou .png)
    if not image.endswith((".jpg", ".png", ".jpeg", ".jfif", ".JPG")):
        return "Le fichier n'est pas une image valide"

    img = Image.open(image)

    # on prend la partie basse de l'image
    croped_image = lower_crop_image(img)

    # On redimensionne l'image
    width = 150
    height = 150
    image_redim = croped_image.resize((width, height))

    if representation == 'HC':
        return histo_image(image_redim)

    if representation == 'PX':
        return tensor_image(image_redim)

    if representation == 'GC':
        return graymatrix_image(image_redim)

    return "il y a un problème"  # voir ce qu'on retourne lorsque l'image a un pb



def histo_image(image):
    return image.histogram()


def tensor_image(image):
    image_np = np.array(image)
    tensor = image_np.astype('float32') / 255.0
    return tensor


def graymatrix_image(image):
    gray_image = np.array(image.convert('L'))
    return gray_image



"""
Returns a data structure embedding train images described according to the 
specified representation and associate each image to its label (image_representation tous de la forme de vecteur)
input = imageData
output = imageData where image_representation = a vector
"""
def transform_to_vecteur(imageData):
    max_vector_size = 0
    for image in imageData:
        image_representation = image['representation']
        vector = np.array(image_representation)
        # longueur du tableau image
        # il cherche le plus grand des vecteurs pour adapter les autres images à ce vecteur
        if vector.shape[0] > max_vector_size:
            max_vector_size = vector.shape[0]
        image['representation'] = vector

    # Remplissage des vecteurs plus petits avec des zéros
    for dict in imageData:
        vector = dict['representation']
        if vector.shape[0] < max_vector_size:
            padding = np.zeros(max_vector_size - vector.shape[0])
            dict['representation'] = np.concatenate((vector, padding))

    return imageData


def normalize_representation_dict(samples_data):
    scaler = StandardScaler()
    representations = [d["representation"] for d in samples_data]
    max_len = 1024
    representations_resized = [np.pad(x, (0, max_len - len(x)), 'constant', constant_values=(0,0)) if len(x) < max_len else x[:max_len] for x in representations]
    X_normalized = scaler.fit_transform(representations_resized)
    for i, d in enumerate(samples_data):
        d["representation"] = X_normalized[i]
    return samples_data



def random_crop(image, crop_size):
    width, height = image.size
    crop_width, crop_height = crop_size
    left = random.randint(0, width - crop_width)
    top = random.randint(0, height - crop_height)
    right = left + crop_width
    bottom = top + crop_height
    return image.crop((left, top, right, bottom))

def random_rotate(image, angle_range=(-180,180)):
    angle = random.uniform(angle_range[0], angle_range[1])
    return image.rotate(angle)

def random_zoom(image, zoom_range):
    zoom_factor = random.uniform(*zoom_range)
    w, h = image.size
    new_w = int(w * zoom_factor)
    new_h = int(h * zoom_factor)
    left = random.randint(0, w - new_w)
    top = random.randint(0, h - new_h)
    right = left + new_w
    bottom = top + new_h
    return image.crop((left, top, right, bottom)).resize(image.size)


"""
Returns a data structure embedding train images described according to the 
specified representation and associate each image to its label.
-> Representation can be (to extend) 
'HC': color histogram
'PX': tensor of pixels 
'GC': matrix of gray pixels
other to be defined
input = where are the data, which represenation of the data must be produced ? 
output = a structure (dictionnary ? Matrix ? File ?) where the images of the
directory have been transformed and labelled according to the directory they are
stored in.
-- uses function raw_image_to_representation
"""
def load_transform_label_train_data(directory, representation):
    image_data = []
    # sous-dossiers contenus dans "directory"
    directories = [
        d for d in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, d))
    ]
    # classifications des images en parcourant chaque sous-dossier
    for etiquette in directories:
        # on accède à chaque sous-dossier de directory
        subdir = os.path.join(directory, etiquette)
        # labelisation et le stockage des images issues du sous-répertoire "Ailleurs" par -1
        if etiquette == "Ailleurs":
            for image in os.listdir(subdir):
                # on transforme l'image par sa representation (type indiqué en paramètre )
                image_representation = raw_image_to_representation(
                    os.path.join(subdir, image), representation)
                image_data.append({'nom': image, 'label': -1, 'representation': image_representation})

        # labelisation des images issues du sous-répertoire "Mer" par 1
        elif etiquette == "Mer":
            for image in os.listdir(subdir):
                # on transforme chaque image par la representation prise en paramètre
                image_representation = raw_image_to_representation(
                    os.path.join(subdir, image), representation)

                # On ajoute un dictionnaire contenant les informations de l'image
                image_data.append({'nom': image, 'label': 1, 'representation': image_representation})

    return image_data


"""
Returns a data structure embedding train images described according to the 
specified representation and associate each image to its label.
-> Representation can be (to extend) 
'HC': color histogram
'PX': tensor of pixels 
'GC': matrix of gray pixels
other to be defined
input = where are the data, which represenation of the data must be produced ? 
output = a structure (dictionnary ? Matrix ? File ?) where the images of the
directory have been croped, transformed and labelled according to the directory they are
stored in.
-- uses function raw_croped_image_to_representation
"""
def load_transform_label_train_data_croped(directory, representation):
    image_data = []
    # sous-dossiers contenus dans "directory"
    directories = [
        d for d in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, d))
    ]
    # classifications des images en parcourant chaque sous-dossier
    for etiquette in directories:
        # on accède à chaque sous-dossier de directory
        subdir = os.path.join(directory, etiquette)
        # labelisation et le stockage des images issues du sous-répertoire "Ailleurs" par -1
        if etiquette == "Ailleurs":
            for image in os.listdir(subdir):
                # on transforme l'image par sa representation (type indiqué en paramètre )
                image_representation = raw_croped_image_to_representation(
                    os.path.join(subdir, image), representation)
                image_data.append({'nom': image, 'label': -1, 'representation': image_representation})
        # labelisation des images issues du sous-répertoire "Mer" par 1
        elif etiquette == "Mer":
            for image in os.listdir(subdir):
                # on transforme chaque image par la representation prise en paramètre
                image_representation = raw_croped_image_to_representation(
                    os.path.join(subdir, image), representation)
                # On ajoute un dictionnaire contenant les informations de l'image
                image_data.append({'nom': image, 'label': 1, 'representation': image_representation})

    return image_data





def load_transform_label_train_data_crop(directory, representation):
    image_data = []

    # sous-dossiers contenus dans "directory"
    directories = [
        d for d in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, d))
    ]

    # classifications des images en parcourant chaque sous-dossier
    for etiquette in directories:
        # on accède à chaque sous-dossier de directory
        subdir = os.path.join(directory, etiquette)
        # labelisation et le stockage des images issues du sous-répertoire "Ailleurs" par -1
        if etiquette == "Ailleurs":
            for image in os.listdir(subdir):
                img_path = os.path.join(subdir, image)
                img = Image.open(img_path)
                suffix=9
                for i in range(3):
                    w, h = img.size
                    if h >= 150 and w >= 150:
                        # On applique un crop aléatoire à chaque image
                        cropped_img = random_crop(img, (150, 150))
                        # On transforme chaque image croppée par sa représentation
                        image_representation = histo_image(cropped_img)
                        # On ajoute un dictionnaire contenant les informations de l'image
                        image_data.append({'nom': f'{os.path.splitext(image)[0]}_crop_{suffix:02d}.jpg', 'label': -1, 'representation': image_representation})
                        suffix+=1
        # labelisation des images issues du sous-répertoire "Mer" par 1
        elif etiquette == "Mer":
            for image in os.listdir(subdir):
                img_path = os.path.join(subdir, image)
                img = Image.open(img_path)
                suffix=9
                for i in range(3):
                    w, h = img.size
                    if h >= 150 and w >= 150:
                        # On applique un crop aléatoire à chaque image
                        cropped_img = random_crop(img, (150, 150))
                        # On transforme chaque image croppée par sa représentation
                        image_representation = histo_image(cropped_img)
                        # On ajoute un dictionnaire contenant les informations de l'image
                        image_data.append({'nom':f'{os.path.splitext(image)[0]}_crop_{suffix:02d}.jpg', 'label': 1, 'representation': image_representation})
                        suffix+=1
    return image_data

def load_transform_label_train_data_rotations(directory, representation):
    image_data = []

    # sous-dossiers contenus dans "directory"
    directories = [
        d for d in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, d))
    ]

    # classifications des images en parcourant chaque sous-dossier
    for etiquette in directories:
        # on accède à chaque sous-dossier de directory
        subdir = os.path.join(directory, etiquette)
        # labelisation et le stockage des images issues du sous-répertoire "Ailleurs" par -1
        if etiquette == "Ailleurs":
            for image in os.listdir(subdir):
                img_path = os.path.join(subdir, image)
                img = Image.open(img_path)
                suffix=1
                for i in range(3):
                    # On effectue une rotation aléatoire sur chaque image
                    rotated_img = random_rotate(img)
                    # On transforme chaque image rotatée par sa représentation
                    image_representation = histo_image(rotated_img)
                    # On ajoute un dictionnaire contenant les informations de l'image
                    image_data.append({'nom':f'{os.path.splitext(image)[0]}_crop_{suffix:02d}.jpg', 'label': -1, 'representation': image_representation})
                    suffix+=1
        # labelisation des images issues du sous-répertoire "Mer" par 1
        elif etiquette == "Mer":
            for image in os.listdir(subdir):
                img_path = os.path.join(subdir, image)
                img = Image.open(img_path)
                suffix=1
                for i in range(3):
                    # On effectue une rotation aléatoire sur chaque image
                    rotated_img = random_rotate(img)
                    # On transforme chaque image rotatée par sa représentation
                    image_representation = histo_image(rotated_img)
                    # On ajoute un dictionnaire contenant les informations de l'image
                    image_data.append({'nom':f'{os.path.splitext(image)[0]}_crop_{suffix:02d}.jpg', 'label': 1, 'representation': image_representation})
                    suffix+=1
    return image_data

def load_transform_label_train_data_zoom(directory, representation):
    image_data = []

    # sous-dossiers contenus dans "directory"
    directories = [
        d for d in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, d))
    ]

    # classifications des images en parcourant chaque sous-dossier
    for etiquette in directories:
        # on accède à chaque sous-dossier de directory
        subdir = os.path.join(directory, etiquette)
        # labelisation et le stockage des images issues du sous-répertoire "Ailleurs" par -1
        if etiquette == "Ailleurs":
            for image in os.listdir(subdir):
                img_path = os.path.join(subdir, image)
                img = Image.open(img_path)
                suffix=5
                for i in range(3):
                    # On effectue un zoom aléatoire sur chaque image
                    zoomed_img = random_zoom(img, (0.2,0.7))
                    # On transforme chaque image zoomée par sa représentation
                    image_representation = histo_image(zoomed_img)
                    # On ajoute un dictionnaire contenant les informations de l'image
                    image_data.append({'nom':f'{os.path.splitext(image)[0]}_crop_{suffix:02d}.jpg', 'label': -1, 'representation': image_representation})
                    suffix+=1
        # labelisation des images issues du sous-répertoire "Mer" par 1
        elif etiquette == "Mer":
            for image in os.listdir(subdir):
                img_path = os.path.join(subdir, image)
                img = Image.open(img_path)
                suffix=5
                for i in range(3):
                    # On effectue un zoom aléatoire sur chaque image
                    zoomed_img = random_zoom(img, (0.2,0.7))
                    # On transforme chaque image zoomée par sa représentation
                    image_representation = histo_image(zoomed_img)
                    # On ajoute un dictionnaire contenant les informations de l'image
                    image_data.append({'nom':f'{os.path.splitext(image)[0]}_crop_{suffix:02d}.jpg', 'label': 1, 'representation': image_representation})
                    suffix+=1
    return image_data








"""
Returns a data structure embedding test images described according to the 
specified representation.
-> Representation can be (to extend) 
'HC': color histogram
'PX': tensor of pixels 
'GC': matrix of gray pixels 
other to be defined
input = where are the data, which represenation of the data must be produced ? 
output = a structure (dictionnary ? Matrix ? File ?) where the images of the directory have been transformed (but not labelled)
-- uses function raw_image_to_representation
"""
# liste où l'on va stocker la représentation de chaque image située dans directory de la forme : samples_data = [dicoImage1 = {"nom": ....,"représentation": .....}; dicoImage2 = {}]
def load_transform_test_data(directory, representation):
    samples_data = []
    # Récupération de la liste des fichiers dans le dossier
    files = os.listdir(directory)

    # Boucle pour parcourir toutes les images
    for image in files:
        # Vérification que le fichier est une image (en utilisant une extension d'image telle que .jpg ou .png)
        if image.endswith(".jpeg") or image.endswith(".png") or image.endswith(".jfif") or image.endswith(".JPG") or image.endswith(".jpg"):
            # recupération de la représentation de l'image qu'on ajoute par la suite à notre liste imagesDirectory
            image_representation = raw_image_to_representation(os.path.join(directory, image), representation)

            samples_data.append({'nom': image, 'representation': image_representation})

    return samples_data



"""
Returns a data structure embedding test images described according to the 
specified representation.
-> Representation can be (to extend) 
'HC': color histogram
'PX': tensor of pixels 
'GC': matrix of gray pixels 
other to be defined
input = where are the data, which represenation of the data must be produced ? 
output = a structure (dictionnary ? Matrix ? File ?) where the images of the directory have been croped and transformed (but not labelled)
-- uses function raw_image_to_representation
"""
def load_transform_test_data_croped(directory, representation):
    samples_data = []
    # Récupération de la liste des fichiers dans le dossier
    files = os.listdir(directory)

    # Boucle pour parcourir toutes les images
    for image in files:
        # Vérification que le fichier est une image (en utilisant une extension d'image telle que .jpg ou .png)
        if image.endswith(".jpeg") or image.endswith(".png") or image.endswith(".jfif") or image.endswith(".JPG") or image.endswith(".jpg"):
            # recupération de la représentation de l'image qu'on ajoute par la suite à notre liste imagesDirectory
            image_representation = raw_croped_image_to_representation(os.path.join(directory, image), representation)

            samples_data.append({'nom': image, 'representation': image_representation})

    return samples_data


"""
Learn a model (function) from a representation of data, using the algorithm 
and its hyper-parameters described in algo_dico
Here data has been previously transformed to the representation used to learn
the model
input = transformed labelled data, the used learning algo and its hyper-parameters (a dico ?)
output =  a model fit with data
"""
def learn_model_from_data(train_data, model):
    X_train = []
    Y_train = []

    for image in train_data:
        X_train.append(image['representation'])
        Y_train.append(image['label'])

    features_array = np.array([np.array(f) for f in X_train])
    labels_array = np.array(Y_train)

    return model.fit(features_array, labels_array)


"""
Given one example (representation of an image as used to compute the model),
computes its class according to a previously learned model.
Here data has been previously transformed to the representation used to learn
the model
input = representation of one data, the learned model
output = the label of that one data (+1 or -1)
-- uses the model learned by function learn_model_from_datas
"""
# si on suppose que data est de cette forme data = [dicoImage1 = {"nom": ....,"représentation": .....}; dicoImage2 = {}]
def predict_example_label(example, model):
    example = example.reshape(1, -1)
    prediction = model.predict(example)
    if prediction > 0:
        return 1
    else:
        return -1


"""
Computes an array (or list or dico or whatever) that associates a prediction 
to each example (image) of the data, using a previously learned model. 
Here data has been previously transformed to the representation used to learn
the model
input = a structure (dico, matrix, ...) embedding all transformed data to a representation, and a model
output =  a structure that associates a label to each data (image) of the input sample
"""
def predict_sample_label(data, model):
    for image in data:
        representation = image["representation"]
        predicted_label = predict_example_label(representation, model)
        image['label'] = predicted_label

    return data



def predict_sample_label_2(data, model):
    names = []
    representations = []
    predicted_labels = []

    for image in data:
        name = image["nom"]
        representation = image["representation"]
        names.append(name)
        representations.append(representation)
        predicted_label = predict_example_label(representation, model)
        predicted_labels.append(predicted_label)
    result = []

    for i in range(len(data)):
        image_dict = {"nom": names[i], "label": predicted_labels[i]}  # ne  renvoie  pas la représentation
        result.append(image_dict)

    return result

"""
Save the predictions on data to a text file with syntax:
filename <space> label (either -1 or 1)  
NO ACCENT  
Here data has been previously transformed to the representation used to learn
the model
input = where to save the predictions, structure embedding the data, the model used
for predictions
output =  OK if the file has been saved, not OK if not
"""
def write_predictions(directory, dataPredicted, newNameFile):
    # construit le chemin du fichier
    filepath = os.path.join(directory, newNameFile)

    # vérifie que dataPredicted n'est pas vide
    if not dataPredicted:
        return "dataPredicted est vide"

    # vérifie que le répertoire existe
    if not os.path.isdir(directory):
        return "répertoire non trouvé"

    # écrit les prédictions dans un fichier texte
    with open(filepath, "w") as file:
        for image in dataPredicted:
            if "nom" in image and "label" in image and image["nom"] is not None and image["label"] is not None:
                file.write(f"{image['nom']} {image['label']}\n")
            else:
                return "données manquantes"

    # vérifie si le fichier a été sauvegardé avec succès
    if os.path.exists(filepath):
        return file
    else:
        return "erreur lors de l'écriture du fichier"



"""
Estimates the accuracy of a previously learned model using train data, 
either through CV or mean hold-out, with k folds.
Here data has been previously transformed to the representation used to learn
the model
input = the train labelled data as previously structured, the learned model, and
the number of split to be used either in a hold-out or by cross-validation  
output =  The score of success (betwwen 0 and 1, the higher the better, scores under 0.5
are worst than random
"""
def estimate_model_score(model, train_data, k):
    learnedModel = learn_model_from_data(train_data, model)
    # récupération des données d'entraînement sous forme de liste (X_train), et des étiquettes associées à chaque données (Y_train)
    X_train = []
    Y_train = []
    for image in train_data:
        X_train.append(image["representation"])
        Y_train.append(image["label"])

    # Utilisation de la méthode cross_val_score pour estimer la performance du modèle en utilisant la validation croisée
    scores = cross_val_score(learnedModel, X_train, Y_train, cv=k)

    # Calcul de la moyenne des scores pour chaque pli
    return scores.mean()





"""
Permet d'enregistrer un modèle appris sous format .pkl
"""
def saveModel(model, nom):
    folder_name = 'Models'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    namePKL = os.path.join(folder_name, nom + '.pkl')
    joblib.dump(model, namePKL)


"""Permet de charger un modèl enregistrer au préalable .pkl"""
def loadLearnedModel(fileModel):
    folder_name = 'Models'
    file_path = os.path.join(folder_name, fileModel)
    # Load the model from the file
    model_from_joblib = joblib.load(file_path)
    return model_from_joblib


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



def calculate_accuracy(predicted_labels, train_data):
    correct_predictions = 0
    total_predictions = 0
    for prediction in predicted_labels:
        for data in train_data:
            if prediction['nom'].startswith(data['nom']):
                if prediction['label'] == data['label']:
                    correct_predictions += 1
                total_predictions += 1
                break  # on a trouvé une correspondance, on passe à l'image suivante
    if total_predictions == 0:
        return 0.0
    return float(correct_predictions) / total_predictions



    # def get_X_y(data_dico):
    #     X = []
    #     y = []
    #     for image in data_dico:
    #         X.append(image['representation'])
    #         y.append(image['label'])
    #     return X, y



