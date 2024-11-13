# File: ClothingDataset.py
# coding=utf-8
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ClothingDataset(Dataset):
    """
    A PyTorch Dataset for loading the clothing dataset for few-shot learning.
    """

    def __init__(self, mode='all_data', root='C:\\work\\few_shot_clothing_detection\\data\\clean_data', transform=None):
        """
        Args:
        - mode: The subset of the dataset to load ('all_data', 'top_10', 'top_100').
        - root: Root directory of the dataset.
        - transform: A function/transform to apply to the images.
        """
        super(ClothingDataset, self).__init__()
        self.mode = mode
        self.root = os.path.join(root, mode)

        # Default transform if none is provided
        self.transform = transform or transforms.Compose([
            transforms.Resize((128, 128)),  # Resize all images to 128x128
            transforms.ToTensor(),         # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
        ])

        if not os.path.exists(self.root):
            raise RuntimeError(f"Dataset folder '{self.root}' not found. Please ensure the path is correct.")

        # Find all items in the dataset (image paths and labels)
        self.samples = self._find_samples()

        # Create a mapping from class names to indices
        self.classes = sorted({label for _, label in self.samples})
        self.class_to_idx = {label: idx for idx, label in enumerate(self.classes)}

        # Convert labels to numeric indices
        self.targets = [self.class_to_idx[label] for _, label in self.samples]

    def _find_samples(self):
        """
        Finds all image files in the dataset folder and their associated labels.
        """
        samples = []
        for root, _, files in os.walk(self.root):
            label = os.path.basename(root)
            for file in files:
                if file.endswith(('.jpg', '.png')):  # Acceptable image formats
                    samples.append((os.path.join(root, file), label))
        return samples

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.samples)

    def __getitem__(self, index):
        """
        Loads and returns a sample from the dataset at the specified index.
        Returns:
        - An image tensor and its corresponding label index.
        """
        path, label = self.samples[index]
        image = Image.open(path).convert('RGB')  # Ensure all images are in RGB format

        if self.transform:
            image = self.transform(image)

        label_idx = self.class_to_idx[label]
        return image, label_idx
