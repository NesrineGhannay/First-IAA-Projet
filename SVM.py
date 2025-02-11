import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageOps
import os
import json
import pickle
from skimage.transform import rescale, resize, downscale_local_mean
from sklearn import svm, model_selection
from sklearn.model_selection import train_test_split
from skimage.feature import local_binary_pattern
import random
from UsualFunctions import *

crop_size = (50, 50)
zoom_range = (0.8, 1.2)
rotations = [0, 90, 180, 270]
flip = True

"""
@authors : Gaël and Simon
Input : a gray matrix, as well as two optional parameters points and radius which define respectively the number of points and the radius of the neighborhood used to extract the LBP (Local Binary Pattern) features.
Output : a normalized histogram obtained by first computing a histogram of the extracted LBP values with the local_binary_pattern function, and normalizing this histogram by dividing it by the sum of its elements.
"""
def local_binary_pattern_features(gray_image, points=24, radius=3):
    lbp = local_binary_pattern(gray_image, points, radius, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, points + 3), range=(0, points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

"""
@autors : Gaël and Simon
Input : an image and a crop size as a tuple (crop_width, crop_height).
Output : a randomly cropped version of the input image, according to the cropping dimensions specified in the input.
"""
def random_crop(image, crop_size):
    width, height = image.size
    crop_width, crop_height = crop_size
    max_x = max(width - crop_width, 0)
    max_y = max(height - crop_height, 0)
    x = random.randint(0, max_x)
    y = random.randint(0, max_y)
    return image.crop((x, y, x + crop_width, y + crop_height))

"""
@autors : Gaël and Simon
Input : an image.
Output : the ratio of blue pixels (with a value greater than 100) in the image as a one-dimensional numpy array.
"""
def blue_ratio(image):
    blue_channel = np.array(image)[:, :, 2]
    total_pixels = blue_channel.size
    blue_pixels = np.sum(blue_channel >100)
    blue_ratio = blue_pixels / total_pixels
    return np.array([blue_ratio])

"""
@autors : Gaël and Simon
Input : an image and several transformation parameters: rotations, flip, crop_size and zoom_range. 
Output : a list of augmented images, which is the combination of all transformations applied to the input image.
"""
def augment_image(image, rotations, flip, crop_size=None, zoom_range=None):
    augmented_images = []
    for angle in rotations:
        rotated_image = image.rotate(angle)
        augmented_images.append(rotated_image)

        if flip:
            flipped_image = ImageOps.mirror(rotated_image)
            augmented_images.append(flipped_image)

        if crop_size is not None:
            cropped_image = random_crop(rotated_image, crop_size)
            augmented_images.append(cropped_image)

            if flip:
                flipped_cropped_image = ImageOps.mirror(cropped_image)
                augmented_images.append(flipped_cropped_image)

        if zoom_range is not None:
            zoom_factor = random.uniform(zoom_range[0], zoom_range[1])
            zoomed_image = image.resize((int(image.width * zoom_factor), int(image.height * zoom_factor)))
            zoomed_cropped_image = random_crop(zoomed_image, (image.width, image.height))
            augmented_images.append(zoomed_cropped_image)

            if flip:
                flipped_zoomed_cropped_image = ImageOps.mirror(zoomed_cropped_image)
                augmented_images.append(flipped_zoomed_cropped_image)
    return augmented_images

"""
@autors : Gaël and Simon
Input : the path to an image. 
Output : the resized image (200, 200).
"""
def process_image(image_path):
    image = Image.open(image_path)
    resized_image = image.resize((200, 200))
    return resized_image

"""
@autors : Gaël and Simon
Input : two images: color_image is the color image and gray_image is the grayscale image.
Output : a feature vector that combines the proportion of blue pixels and the LBP features of the grayscale image.
"""
def extract_features(color_image, gray_image):
    if color_image.mode == "RGB":  # Vérifie que l'image est en couleur
        blue_feature = blue_ratio(color_image)
    else:
        blue_feature = np.array([0])  # Aucun pixel bleu dans une image en niveaux de gris

    lbp_feature = local_binary_pattern_features(gray_image)

    feature_vector = np.concatenate((blue_feature, lbp_feature))
    return feature_vector

"""
@autors : Gaël and Simon
Input : a directory containing the training images.
Output : a list of elements, each representing a transformed and labeled image. Each element contains the name of the image, its label (1 for "Sea" and -1 for "Elsewhere") and its representation as a feature vector.
"""
def load_transform_label_train_data_svm(directory):
    image_data = []
    label_dirs = {'Ailleurs': -1, 'Mer': 1}

    for label, value in label_dirs.items():
        subdir = os.path.join(directory, label)

        for image in os.listdir(subdir):
            img_path = os.path.join(subdir, image)
            img_orig = Image.open(img_path)
            img_resized = img_orig.resize((200, 200))
            augmented_images = augment_image(img_resized, rotations, flip, crop_size=crop_size, zoom_range=zoom_range)

            for aug_image in augmented_images:
                gray_aug_image = aug_image.convert('L')
                feature_vector = extract_features(img_orig, gray_aug_image)
                image_data.append({'nom': image, 'label': value, 'representation': feature_vector})

    return image_data

"""
@autors : Gaël and Simon
Input : a trained classifier (clf), a list of features (X_test) and labels (y_test) for a test data set, and the corresponding original image data (image_data).
Output : just displays the results on the screen.
"""
def predict_and_display(clf, X_test, y_test, image_data):
    predictions = clf.predict(X_test)
    correct_predictions = predictions == y_test

    for i, prediction in enumerate(predictions):
        image_name = image_data[i]['nom']
        true_label = y_test[i]
        is_correct = correct_predictions[i]

        print(f"Image: {image_name}")
        print(f"Prédiction: {'Mer' if prediction == 1 else 'Ailleurs'}")
        print(f"Vrai label: {'Mer' if true_label == 1 else 'Ailleurs'}")
        print(f"Prédiction correcte: {is_correct}")
        print()

"""
@autors : Gaël and Simon
Input : a directory containing the data to train the model using SVM.
Output : the trained SVM model (finalModel).
"""
def SVM_model(filedata):
    image_data = load_transform_label_train_data_svm(filedata)
    X, y = get_features_array(image_data)
    svm_model = svm.SVC(kernel='linear', C=1)
    finalModel = learn_model_from_data(image_data, svm_model)
    return finalModel

"""
@autors : Gaël and Simon
Input : a dictionary containing transformed and labeled image data
Output : two numpy arrays, one containing the features extracted from the images and the other containing the associated labels.
"""
def get_features_array(dico_data):
    features_array = [d['representation'] for d in dico_data]
    labels_array = [d['label'] for d in dico_data]
    return features_array, labels_array

"""
@autors : Gaël and Simon
Input : a directory containing the data to train the model using SVM, a model that will be saved now.
Output : nothing.
"""
def saveSVM(filedata):
    model = SVM_model(filedata)
    saveModel(model, "SVM")

#saveSVM("Data")

"""
@autors : Gaël and Simon
Input : the path to a directory containing images to be tested.
Output : list of dictionaries, each containing the name of an image and its representation as a feature vector extracted from this image.
"""
def load_test_data_svm(fileTestData):
    test_data = []
    for image in os.listdir(fileTestData):
        img_path = os.path.join(fileTestData, image)
        color_image = process_image(img_path)
        gray_image = color_image.convert('L')
        feature_vector = extract_features(color_image, gray_image)
        test_data.append({'nom': image, 'representation': feature_vector})
    return test_data

"""
@autors : Gaël and Simon
Input : a directory containing test images (fileTestData) and a trained SVM model (model).
Output : a list containing dictionaries for each image, with the keys 'name' (the name of the image file) and 'label' (the model prediction).
"""
def predict_with_SVM(fileTestData, model):
    testData = load_test_data_svm(fileTestData)
    for data in testData:
        image_name = data['nom']
        feature_vector = data['representation']
        prediction = model.predict([feature_vector])[0]
        data['label'] = prediction
    return testData

"""
@autors : Gaël and Simon
Input : three arguments as input: fileModel(the path to the saved trained SVM model), fileTestData(the path to the directory containing test images), and fileForPredictedData (the path to the file where the predicted labels for the test images will be saved). 
Output : only writes the predicted labels to a file.
"""
def mainSVM_prediction(fileModel, fileTestData, fileForPredictedData):
    svmModel = loadLearnedModel(fileModel)
    predictedData = predict_with_SVM(fileTestData, svmModel)
    write_predictions("Predictions", predictedData, fileForPredictedData)

mainSVM_prediction("SVM.pkl", "TestCC2", "PredictionsCC2_SVM.txt")

'''
@author: Nesrine
Allows to keep the success percentage of SVM model
input: the learning file
output: a percentage representing the learning rate of the model taken as a parameter
function used: loadLearnedModel(), load_transform_label_train_data_svm() & estimate_model_score()
'''
def estimate_SVM_score(fileData):
    model = loadLearnedModel("SVM.pkl")
    data = load_transform_label_train_data_svm(fileData)
    return estimate_model_score(model, data, 10)

# print(estimate_SVM_score("Data"))