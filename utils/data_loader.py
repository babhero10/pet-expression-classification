from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import random

def load_data(data_dir, img_size=(224, 224), batch_size=32, num_augmentations=3, seed=42):
    """
    Loads image data and augments the training set dynamically without saving.

    Args:
        data_dir (str): Path to the directory containing 'train', 'valid', and 'test' folders.
        img_size (tuple): Desired image size (height, width).
        batch_size (int): Batch size for data loaders.
        num_augmentations (int): Number of augmented versions per image to add to the training set.
        seed (int): Seed for reproducible shuffling.
    Returns:
        tuple: train_loader, val_loader, test_loader, class_labels
    """
    random.seed(seed)

    class CustomDataset(Dataset):
        def __init__(self, root_dir, transform=None):
            self.root_dir = root_dir
            self.transform = transform
            self.image_paths = []
            self.labels = []
            self.classes = sorted(os.listdir(root_dir))
            for label, class_name in enumerate(self.classes):
                class_dir = os.path.join(root_dir, class_name)
                for img_name in os.listdir(class_dir):
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(label)

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            img_path = self.image_paths[idx]
            label = self.labels[idx]
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, label

    class AugmentedDataset(Dataset):
        def __init__(self, base_dataset, transform, num_augmentations):
            self.base_dataset = base_dataset
            self.transform = transform
            self.num_augmentations = num_augmentations

        def __len__(self):
            return len(self.base_dataset) * self.num_augmentations

        def __getitem__(self, idx):
            # Map idx to the original dataset and augmentation index
            original_idx = idx // self.num_augmentations
            _, label = self.base_dataset[original_idx]  # Get the label from the original dataset
            img_path = self.base_dataset.image_paths[original_idx]
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, label

    # Original and augmented transforms
    original_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    augmentation_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(
            degrees=15,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            shear=10
        ),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Datasets
    train_dataset = CustomDataset(os.path.join(data_dir, "train"), transform=original_transform)
    augmented_dataset = AugmentedDataset(train_dataset, transform=augmentation_transform, num_augmentations=num_augmentations)

    # Combine original and augmented datasets
    combined_train_dataset = ConcatDataset([train_dataset, augmented_dataset])

    val_dataset = CustomDataset(os.path.join(data_dir, "valid"), transform=original_transform)
    test_dataset = CustomDataset(os.path.join(data_dir, "test"), transform=original_transform)

    # DataLoaders
    train_loader = DataLoader(combined_train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_dataset.classes
