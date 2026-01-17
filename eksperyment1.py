import sys

import cv2
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
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
reduced_features = apply_pca(features, n_components=600)
print(f"reduced {reduced_features.shape}")
# Combine reduced features with labels and save the dataset
dataset = np.column_stack((reduced_features, labels))

sampling = ""
sampler = None
X = reduced_features
y = np.array(labels)
path = sys.argv[1]
print(sys.argv[2])
if sys.argv[2] == "SMOTE":
    # print("Applying SMOTE")
    sampler = SMOTE(random_state=42)
    # X, y = smote.fit_resample(X, y)
    sampling = "smote"
elif sys.argv[2] == "ROS":
    # print("Applying ROS")
    sampler = RandomOverSampler(random_state=42)
    # X, y = ros.fit_resample(X, y)
    sampling = "ros"
elif sys.argv[2] == "RUS":
    # print("Applying RUS")
    sampler = RandomUnderSampler(random_state=42)
    # X, y = rus.fit_resample(X, y)
    sampling = "rus"
elif sys.argv[2] == "CNN":
    # print("Applying CNN")
    sampler = CondensedNearestNeighbour()
    # X, y = cnn.fit_resample(X, y)
    sampling = "cnn"
else:
    sampling = NO_SAMPLING

rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=36851234)

svm_results = ResultHelper(f"SVC_{sampling}", path)
kNN_results = ResultHelper(f"kNN_{sampling}", path)
nB_results = ResultHelper(f"NB_{sampling}", path)
logistic_results = ResultHelper(f"LR_{sampling}", path)

iteration = 0

for train_index, test_index in rskf.split(X, y):
    print(f"iteration = {iteration}")
    iteration = iteration + 1

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    if sampling != NO_SAMPLING:
        print(f"Applying {sampling}")
        X_train, y_train = sampler.fit_resample(X_train, y_train)
        if (iteration == 1):
            plot_class_distribution(y_train, folder_path=path)

    print("Training SVC")
    svm_classifier = SVC(kernel='linear')
    svm_classifier.fit(X_train, y_train)
    y_pred = svm_classifier.predict(X_test)

    majority_class_count = np.sum(y_pred == 0)
    majority_class_percentage = (majority_class_count / len(y_pred)) * 100
    print(f"Percentage of predictions belonging to the majority class: {majority_class_percentage:.2f}%")

    svm_results.append_all_scores(y_test, y_pred)

    print("Training kNN")
    kNN_classifier = KNeighborsClassifier(n_neighbors=3)
    kNN_classifier.fit(X_train, y_train)
    y_pred = kNN_classifier.predict(X_test)

    majority_class_count = np.sum(y_pred == 0)
    majority_class_percentage = (majority_class_count / len(y_pred)) * 100
    print(f"Percentage of predictions belonging to the majority class: {majority_class_percentage:.2f}%")

    kNN_results.append_all_scores(y_test, y_pred)

    print("Training NB")
    nB_classifier = GaussianNB()
    nB_classifier.fit(X_train, y_train)
    y_pred = nB_classifier.predict(X_test)

    majority_class_count = np.sum(y_pred == 0)
    majority_class_percentage = (majority_class_count / len(y_pred)) * 100
    print(f"Percentage of predictions belonging to the majority class: {majority_class_percentage:.2f}%")

    nB_results.append_all_scores(y_test, y_pred)

    print("Training logistic regression")
    logistic_classifier = LogisticRegression(max_iter=1000)
    logistic_classifier.fit(X_train, y_train)
    y_pred = logistic_classifier.predict(X_test)

    majority_class_count = np.sum(y_pred == 0)
    majority_class_percentage = (majority_class_count / len(y_pred)) * 100
    print(f"Percentage of predictions belonging to the majority class: {majority_class_percentage:.2f}%")

    logistic_results.append_all_scores(y_test, y_pred)


svm_results.plot_radar_chart_for_ml_models()
svm_results.save_scores()

kNN_results.plot_radar_chart_for_ml_models()
kNN_results.save_scores()

nB_results.plot_radar_chart_for_ml_models()
nB_results.save_scores()

logistic_results.plot_radar_chart_for_ml_models()
logistic_results.save_scores()


models = ({
    "SVC": svm_results.scores,
    "kNN": kNN_results.scores,
    "NB": nB_results.scores,
    "RL": logistic_results.scores
    })

ResultHelper.plot_radar_combined(models, path)
