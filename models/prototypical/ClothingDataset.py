# coding=utf-8
from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import numpy as np
import torch
import os

IMG_CACHE = {}

class ClothingDataset(data.Dataset):
    def __init__(self, mode='all_data', root='C:\work\few_shot_clothing_detection\data\clean_data', 
                 transform=None, target_transform=None):
        '''
        Args:
        - mode: which dataset folder to use ('all_data', 'top_10', 'top_100')
        - root: the directory where the dataset is stored
        - transform: how to transform the input
        - target_transform: how to transform the target
        '''
        super(ClothingDataset, self).__init__()
        self.root = os.path.join(root, mode)
        self.transform = transform
        self.target_transform = target_transform

        if not os.path.exists(self.root):
            raise RuntimeError(f"Dataset folder '{self.root}' not found. Please ensure the path is correct.")

        # Gather all items and classes
        self.all_items = self.find_items(self.root)
        self.idx_classes = self.index_classes(self.all_items)

        # Extract file paths and labels
        self.paths, self.y = zip(*[self.get_path_label(idx) for idx in range(len(self))])

        # Load all images into memory
        self.x = list(map(self.load_img, self.paths))

    def __getitem__(self, idx):
        x = self.x[idx]
        if self.transform:
            x = self.transform(x)
        return x, self.y[idx]

    def __len__(self):
        return len(self.all_items)

    def get_path_label(self, index):
        file_path, label = self.all_items[index]
        target = self.idx_classes[label]
        if self.target_transform:
            target = self.target_transform(target)
        return file_path, target

    def find_items(self, root_dir):
        items = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(("jpg", "png")):  # Accept JPG or PNG files
                    label = os.path.basename(root)  # Folder name is the class label
                    items.append((os.path.join(root, file), label))
        print(f"== Dataset: Found {len(items)} items in '{root_dir}'")
        return items

    def index_classes(self, items):
        # Create an index mapping for class names to numeric labels
        class_set = sorted(set(label for _, label in items))
        idx = {class_name: i for i, class_name in enumerate(class_set)}
        print(f"== Dataset: Found {len(idx)} classes")
        return idx

    def load_img(self, path):
        # Load and preprocess an image
        if path in IMG_CACHE:
            img = IMG_CACHE[path]
        else:
            img = Image.open(path).convert("RGB")
            IMG_CACHE[path] = img
        img = img.resize((128, 128))  # Resize to 128x128
        img = np.array(img, np.float32) / 255.0  # Normalize to [0, 1]
        img = torch.from_numpy(img).permute(2, 0, 1)  # Convert to tensor (C, H, W)
        return img
