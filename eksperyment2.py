import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import csv

from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score,
    recall_score, balanced_accuracy_score
)
from imblearn.metrics import geometric_mean_score, specificity_score
from math import pi

from utils.ImageDataset import ImageDataset

# =====================
# CONFIG
# =====================
BALANCED = False
EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-3
RESULTS_DIR = "results/resnet_imb"

os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# METRICS
# =====================
metric_names = ["F1", "REC", "PREC", "ACC", "BAC", "GMEAN", "SPEC"]
all_scores = {k: [] for k in metric_names}

# =====================
# DATASET
# =====================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3)
])

labels_file = "dataset/labels.csv"
labels_df = pd.read_csv(labels_file)
labels = labels_df["sick"].values

dataset = ImageDataset(
    img_dir="dataset/images/",
    labels_file=labels_file,
    transform=transform
)

indices = np.arange(len(labels))

train_idx, val_idx = train_test_split(
    indices,
    test_size=0.2,
    stratify=labels,
    random_state=42
)

train_dataset = Subset(dataset, train_idx)
val_dataset = Subset(dataset, val_idx)

# =====================
# SAMPLING
# =====================
if BALANCED:
    train_labels = labels[train_idx]
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[train_labels]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler
    )
else:
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# =====================
# MODEL
# =====================
model = models.resnet18(pretrained=True)

for p in model.parameters():
    p.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

# =====================
# LOSS & OPTIMIZER
# =====================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LR)

# =====================
# TRAINING LOOP
# =====================
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)

    # =====================
    # VALIDATION
    # =====================
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, 1).cpu().numpy()

            y_pred.extend(preds)
            y_true.extend(targets.numpy())

    # =====================
    # METRICS
    # =====================
    scores = {
        "F1": f1_score(y_true, y_pred),
        "REC": recall_score(y_true, y_pred),
        "PREC": precision_score(y_true, y_pred),
        "ACC": accuracy_score(y_true, y_pred),
        "BAC": balanced_accuracy_score(y_true, y_pred),
        "GMEAN": geometric_mean_score(y_true, y_pred),
        "SPEC": specificity_score(y_true, y_pred)
    }

    for k in all_scores:
        all_scores[k].append(scores[k])

    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Loss: {avg_loss:.4f} "
          f"F1: {scores['F1']:.3f} "
          f"REC: {scores['REC']:.3f}")

# =====================
# RADAR CHART
# =====================
def plot_radar(scores, title, path):
    labels = list(scores.keys())
    values = list(scores.values())
    values += values[:1]

    N = len(labels)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    plt.figure(figsize=(6,6))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1], labels)
    ax.set_rlabel_position(0)
    plt.yticks(np.linspace(0,1,11), fontsize=8)
    plt.ylim(0,1)

    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.25)

    plt.title(title, y=1.1)
    plt.savefig(path, dpi=200)
    plt.show()

# =====================
# FINAL RESULTS
# =====================
mean_scores = {k: np.mean(v) for k, v in all_scores.items()}

mode = "balanced" if BALANCED else "imbalanced"
csv_path = os.path.join(RESULTS_DIR, f"resnet_{mode}_scores.csv")

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch"] + metric_names)

    for i in range(EPOCHS):
        row = [i + 1] + [all_scores[m][i] for m in metric_names]
        writer.writerow(row)

print(f"Saved epoch-wise results to: {csv_path}")
print("\n=== FINAL AVERAGE RESULTS ===")
for k, v in mean_scores.items():
    print(f"{k:6s}: {v:.4f}")

mode = "balanced" if BALANCED else "imbalanced"
plot_radar(
    mean_scores,
    title=f"ResNet18 ({mode})",
    path=f"{RESULTS_DIR}/resnet_{mode}.png"
)
