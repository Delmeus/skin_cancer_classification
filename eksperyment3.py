import sys

import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
import os
from utils.ResultHelper import ResultHelper, plot_class_distribution
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, CondensedNearestNeighbour

NO_SAMPLING = "imbalanced"

if len(sys.argv) < 3:
    print("Not enough arguments")
    sys.exit(1)

def load_images_from_csv(csv_file, img_folder):
    print("Loading images...")
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


csv_file = 'dataset/labels.csv'
img_folder = "dataset/images"
images, labels = load_images_from_csv(csv_file, img_folder)


hog = cv2.HOGDescriptor()


def extract_hog_features(images):
    print("Applying HOG...")
    features = []
    for img in images:
        hog_features = hog.compute(img).flatten()
        features.append(hog_features)
    return np.array(features)


from sklearn.decomposition import PCA

def apply_pca(features, n_components=50):
    print("Applying PCA...")
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(features)
    return reduced_features

# Extract HOG features
features = extract_hog_features(images)
print(f"features {features.shape}")
# Apply PCA to HOG features
reduced_features = apply_pca(features, n_components=1000)
print(f"reduced {reduced_features.shape}")
# Combine reduced features with labels and save the dataset
dataset = np.column_stack((reduced_features, labels))

sampling = NO_SAMPLING
sampler = None
X = reduced_features
y = np.array(labels)
path = sys.argv[1]
if sys.argv[2] == "SMOTE":
    sampler = SMOTE(random_state=263914)
    sampling = "smote"
elif sys.argv[2] == "ROS":
    sampler = RandomOverSampler(random_state=263914)
    sampling = "ros"
elif sys.argv[2] == "RUS":
    sampler = RandomUnderSampler(random_state=263914)
    sampling = "rus"
elif sys.argv[2] == "CNN":
    sampler = CondensedNearestNeighbour()
    sampling = "cnn"

plot_class_distribution(y, folder_path=path)

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=36851234)

ada_results = ResultHelper(f"ADA_{sampling}", path)
wrf_results = ResultHelper(f"WRF_{sampling}", path)

iteration = 1

for train_index, test_index in rskf.split(X, y):
    print(f"iteration = {iteration}")
    iteration = iteration + 1

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    if sampling != NO_SAMPLING:
        print(f"Applying {sampling}")
        X_train, y_train = sampler.fit_resample(X_train, y_train)

    print("Training AdaBoost")
    ada_classifier = AdaBoostClassifier()
    ada_classifier.fit(X_train, y_train)
    y_pred = ada_classifier.predict(X_test)

    majority_class_count = np.sum(y_pred == 0)
    majority_class_percentage = (majority_class_count / len(y_pred)) * 100
    print(f"Percentage of predictions belonging to the majority class: {majority_class_percentage:.2f}%")

    ada_results.append_all_scores(y_test, y_pred)

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    print("Training WRF")
    weight = 20 if sampling == NO_SAMPLING else 1
    wrf_classifier = RandomForestClassifier(class_weight={1: weight, 0: 1})
    wrf_classifier.fit(X_train, y_train)
    y_pred = wrf_classifier.predict(X_test)

    majority_class_count = np.sum(y_pred == 0)
    majority_class_percentage = (majority_class_count / len(y_pred)) * 100
    print(f"Percentage of predictions belonging to the majority class: {majority_class_percentage:.2f}%")

    wrf_results.append_all_scores(y_test, y_pred)


# ada_results.plot_radar_chart_for_ml_models()
ada_results.save_scores()
wrf_results.save_scores()

models = ({
    "AdaBoost": ada_results.scores,
    "Random Forest": wrf_results.scores
    })

ResultHelper.plot_radar_combined(models, path)
