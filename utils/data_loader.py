import os
from torch import manual_seed
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, Subset
from PIL import Image
import random

def load_data(data_dir, img_size=(224, 224), batch_size=32, num_train_samples=None, seed=42):
    """
    Loads image data, optionally limiting the number of training samples.

    Args:
        data_dir (str): Path to the directory containing 'train', 'valid', and 'test' folders.
        img_size (tuple): Desired image size (height, width).
        batch_size (int): Batch size for data loaders.
        num_train_samples (int, optional): If specified, the maximum number of training samples to use. Defaults to None (use all).
        seed (int, optional) : seed for reproducible shuffling.
    Returns:
        tuple: train_loader, val_loader, test_loader, class_labels
    """
    random.seed(seed)
    manual_seed(seed)

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

    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = CustomDataset(os.path.join(data_dir, "train"), transform=transform)
    val_dataset   = CustomDataset(os.path.join(data_dir, "valid"), transform=transform)
    test_dataset  = CustomDataset(os.path.join(data_dir, "test"), transform=transform)

    # Limit training data if num_train_samples is specified
    if num_train_samples is not None:
        indices = list(range(len(train_dataset)))
        random.shuffle(indices)
        train_indices = indices[:num_train_samples]
        train_subset = Subset(train_dataset, train_indices)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    return train_loader, val_loader, test_loader, train_dataset.classes
