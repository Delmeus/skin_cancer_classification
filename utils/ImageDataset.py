import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class ImageDataset(Dataset):
    def __init__(self, img_dir, labels_file, transform=None):
        # Load labels from the CSV file
        self.labels = pd.read_csv(labels_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Get image file name and label
        img_name = os.path.join(self.img_dir, self.labels.iloc[idx, 0]) + '.jpg'
        img_name = os.path.normpath(img_name)
        label = self.labels.iloc[idx, 1]

        # Load the image
        image = Image.open(img_name).convert("RGB")

        # Apply transformations (if any)
        if self.transform:
            image = self.transform(image)

        return image, label

