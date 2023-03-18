import numpy as np
from sklearn.preprocessing import StandardScaler


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

