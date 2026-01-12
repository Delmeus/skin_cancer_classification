import numpy as np
import cv2
import pandas as pd
import os


def load_images_from_csv(csv_file, img_folder):
    data = pd.read_csv(csv_file)
    images = []
    labels = []

    for index, row in data.iterrows():
        img_path = os.path.join(img_folder, row['image']) + '.jpg'
        img_path = os.path.normpath(img_path)
        label = row['sick']

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.resize(img, (64, 128))
            images.append(img)
            labels.append(label)
        else:
            print("Can't read image!")

    return images, labels


csv_file = '../dataset/labels.csv'
img_folder = '../dataset/images'
images, labels = load_images_from_csv(csv_file, img_folder)

def extract_hog_features(images):
    hog = cv2.HOGDescriptor()
    features = []
    for img in images:
        hog_features = hog.compute(img).flatten()
        features.append(hog_features)
    return np.array(features)

# array = extract_hog_features(images)

from sklearn.decomposition import PCA

def apply_pca(features, n_components=1000):
    """
    Apply PCA to reduce dimensionality of the feature set.
    """
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(features)
    return reduced_features

# Extract HOG features
features = extract_hog_features(images)

# Apply PCA to HOG features
reduced_features = apply_pca(features, n_components=1000)

# Combine reduced features with labels and save the dataset
dataset = np.column_stack((reduced_features, labels))
print(dataset.shape)
np.save('../npy_datasets/hog_dataset_with_pca.npy', dataset)


# np.save('../../../npy_datasets/hog_dataset.npy', dataset)
