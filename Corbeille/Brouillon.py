from UsualFunctions import *


def get_X_y(data_dico):
    X = []
    y = []
    for image in data_dico:
        X.append(image['representation'])
        y.append(image['label'])
    return X, y

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
