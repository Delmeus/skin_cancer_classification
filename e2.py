import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score, balanced_accuracy_score
)
from imblearn.metrics import geometric_mean_score, specificity_score
from imblearn.over_sampling import SMOTE
from utils.ImageDataset import ImageDataset
from scipy.ndimage import gaussian_filter1d
import torchvision.models as models
from sklearn.model_selection import train_test_split

USE_SMOTE = False
BATCH_SIZE = 50
EPOCHS = 30
TEST_SIZE = 0.2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

score_names = ["F1", "Recall", "Precision", "Accuracy",
               "Balanced Accuracy", "G-Mean", "Specificity"]

metric_keys = ["f1", "rec", "prec", "acc", "bac", "gmean", "spec"]

def get_scores(y_true, y_pred):
    return {
        "f1": f1_score(y_true, y_pred),
        "rec": recall_score(y_true, y_pred),
        "prec": precision_score(y_true, y_pred),
        "acc": accuracy_score(y_true, y_pred),
        "bac": balanced_accuracy_score(y_true, y_pred),
        "gmean": geometric_mean_score(y_true, y_pred),
        "spec": specificity_score(y_true, y_pred),
    }

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5)),
])

labels_file = "dataset/labels.csv"
dataset = ImageDataset(
    img_dir='dataset/images/',
    labels_file=labels_file,
    transform=transform
)

labels = pd.read_csv(labels_file)["sick"].values
indices = np.arange(len(dataset))

train_idx, test_idx = train_test_split(
    indices,
    test_size=TEST_SIZE,
    stratify=labels,
    random_state=263914
)

train_loader = DataLoader(
    Subset(dataset, train_idx),
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    Subset(dataset, test_idx),
    batch_size=BATCH_SIZE,
    shuffle=False
)

model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

feature_dim = model.fc.in_features
model.fc = nn.Identity()  # feature extractor
model = model.to(device)

classifier = nn.Linear(feature_dim, 2).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=0.001)

def extract_features(loader):
    model.eval()
    features, targets = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            feats = model(x)
            features.append(feats.cpu().numpy())
            targets.append(y.numpy())

    return np.vstack(features), np.hstack(targets)

X_train, y_train = extract_features(train_loader)

if USE_SMOTE:
    smote = SMOTE(random_state=263914)
    X_train, y_train = smote.fit_resample(X_train, y_train)

X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)

for epoch in range(EPOCHS):
    classifier.train()
    optimizer.zero_grad()

    outputs = classifier(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {loss.item():.4f}")

X_test, y_test = extract_features(test_loader)

classifier.eval()
with torch.no_grad():
    outputs = classifier(torch.tensor(X_test, dtype=torch.float32).to(device))
    _, y_pred = torch.max(outputs, 1)

y_pred = y_pred.cpu().numpy()

scores = get_scores(y_test, y_pred)

print("\n=== AVERAGE TEST RESULTS ===")
for k in metric_keys:
    print(f"{k.upper():>5}: {scores[k]:.4f}")
