import os
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
import random
import numpy as np

class PetExpressionClassificationDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @property
    def classes(self):
        return self.data.classes


def load_data(data_dir, batch_size, model_name, seed=42):
    """Loads and preprocesses the dataset."""
    # Transformations with a consistent seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    img_size = 224
    if model_name == "inception_v3" or model_name == "pretrained":
        img_size = 299

    # Data augmentation and normalization for training
    augmentation_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.01, contrast=0.01, saturation=0.01, hue=0.01),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 0.25)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    original_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load datasets
    train_dataset = PetExpressionClassificationDataset(os.path.join(data_dir, 'train'), transform=augmentation_transform)
    val_dataset = PetExpressionClassificationDataset(os.path.join(data_dir, 'valid'), transform=original_transform)
    test_dataset = PetExpressionClassificationDataset(os.path.join(data_dir, 'test'), transform=original_transform)

    # Prepare dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

    class_labels = train_dataset.classes

    return train_loader, val_loader, test_loader, class_labels
