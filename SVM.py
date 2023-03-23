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



rotations = [0, 90, 180, 270]
flip = True

def local_binary_pattern_features(gray_image, points=24, radius=3):
    lbp = local_binary_pattern(gray_image, points, radius, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, points + 3), range=(0, points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

def random_crop(image, crop_size):
    width, height = image.size
    crop_width, crop_height = crop_size
    max_x = width - crop_width
    max_y = height - crop_height
    x = random.randint(0, max_x)
    y = random.randint(0, max_y)
    return image.crop((x, y, x + crop_width, y + crop_height))


def blue_ratio(image):
    blue_channel = np.array(image)[:, :, 2]
    total_pixels = blue_channel.size
    blue_pixels = np.sum(blue_channel >100)
    blue_ratio = blue_pixels / total_pixels
    return np.array([blue_ratio])

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

def process_image(image_path):
    image = Image.open(image_path)
    resized_image = image.resize((200, 200))
    return resized_image


def extract_features(color_image, gray_image):
    if color_image.mode == "RGB":  # Vérifie que l'image est en couleur
        blue_feature = blue_ratio(color_image)
    else:
        blue_feature = np.array([0])  # Aucun pixel bleu dans une image en niveaux de gris

    lbp_feature = local_binary_pattern_features(gray_image)

    feature_vector = np.concatenate((blue_feature, lbp_feature))
    return feature_vector


def load_transform_label_train_data_svm(directory):
    image_data = []
    label_dirs = {'Ailleurs': -1, 'Mer': 1}

    for label, value in label_dirs.items():
        subdir = os.path.join(directory, label)

        for image in os.listdir(subdir):
            img_path = os.path.join(subdir, image)
            img_orig = Image.open(img_path)
            img_resized = img_orig.resize((200, 200))
            augmented_images = augment_image(img_resized, rotations, flip)

            for aug_image in augmented_images:
                gray_aug_image = aug_image.convert('L')
                feature_vector = extract_features(img_orig, gray_aug_image)
                image_data.append({'nom': image, 'label': value, 'representation': feature_vector})

    return image_data
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

def SVM_model(filedata):
    image_data = load_transform_label_train_data_svm(filedata)
    X, y = get_features_array(image_data)
    svm_model = svm.SVC(kernel='linear', C=1)
    finalModel = learn_model_from_data(image_data, svm_model)
    return finalModel


def get_features_array(dico_data):
    features_array = [d['representation'] for d in dico_data]
    labels_array = [d['label'] for d in dico_data]
    return features_array, labels_array

# Enregistrez le modèle SVM entraîné
def saveSVM(filedata):
    model = SVM_model(filedata)
    saveModel(model, "SVM")
saveSVM("Data")

def load_test_data_svm(fileTestData):
    test_data = []
    for image in os.listdir(fileTestData):
        img_path = os.path.join(fileTestData, image)
        color_image = process_image(img_path)
        gray_image = color_image.convert('L')
        feature_vector = extract_features(color_image, gray_image)
        test_data.append({'nom': image, 'representation': feature_vector})
    return test_data


def predict_with_SVM(fileTestData, model):
    testData = load_test_data_svm(fileTestData)
    for data in testData:
        image_name = data['nom']
        feature_vector = data['representation']
        prediction = model.predict([feature_vector])[0]
        data['label'] = prediction
    return testData


def mainSVM_prediction(fileModel, fileTestData, fileForPredictedData):
    svmModel = loadLearnedModel(fileModel)
    predictedData = predict_with_SVM(fileTestData, svmModel)
    write_predictions("Predictions", predictedData, fileForPredictedData)


