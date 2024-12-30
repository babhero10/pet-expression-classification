import os
import matplotlib.pyplot as plt
import torchvision
import random
from utils.data_loader import load_data

def analyze_dataset(data_dir, model_name, batch_size=32, num_visualized_images=16, num_augmented_images=16, seed=42):
    """Analyzes the dataset, visualizes data, and provides class info for train, val, and test sets."""

    analysis_dir = "analysis"
    os.makedirs(analysis_dir, exist_ok=True)

    # Load data using your existing function
    train_loader, val_loader, test_loader, class_labels = load_data(data_dir, batch_size, model_name, seed=seed)

    datasets = {
        "train": train_loader.dataset,
        "val": val_loader.dataset,
        "test": test_loader.dataset
    }

    for dataset_name, dataset in datasets.items():

        # --- Class Distribution ---
        print(f"\n=== {dataset_name.capitalize()} Class Distribution ===")
        class_counts = {}
        for _, label in dataset:
            if label not in class_counts:
                class_counts[label] = 0
            class_counts[label] += 1

        for i, class_name in enumerate(class_labels):
            count = class_counts.get(i, 0)
            print(f"Class: {class_name}, Count: {count}")

        num_classes = len(class_labels)
        print(f"Total Classes: {num_classes}")

        # Plotting Class Distribution
        plt.figure(figsize=(10, 6))
        bars = plt.bar(class_labels, [class_counts.get(i, 0) for i in range(num_classes)])
        plt.xlabel("Class Names")
        plt.ylabel("Number of Samples")
        plt.title(f"{dataset_name.capitalize()} Class Distribution")
        plt.xticks(rotation=45, ha="right")

        # Add value labels on top of the bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{int(height)}',
                     ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, f"{dataset_name}_class_distribution.png"))
        plt.close()
        print(f"{dataset_name.capitalize()} class distribution plot saved.")

        # --- Visualize Images ---
        print(f"\n=== Saving {dataset_name.capitalize()} Images ===")

        visualized_images = []
        if dataset_name == "train":  # Visualize augmented images only for training set
            for i in range(num_augmented_images):
                image, _ = dataset[random.randint(0, len(dataset) - 1)]
                visualized_images.append(image)

            title = f"{dataset_name.capitalize()} Augmented Images"
        else:  # Visualize original images for validation and test sets
            for i in range(num_visualized_images):
                image, _ = dataset[random.randint(0, len(dataset) - 1)]
                visualized_images.append(image)

            title = f"{dataset_name.capitalize()} Images"

        grid = torchvision.utils.make_grid(visualized_images)

        plt.figure(figsize=(12, 12))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
        plt.axis('off')
        plt.title(title)
        plt.savefig(os.path.join(analysis_dir, f"{dataset_name}_images.png"))
        plt.close()
        print(f"{len(visualized_images)} {dataset_name.capitalize()} images saved.")

    print(f"\nAnalysis complete. Results saved in '{analysis_dir}' folder.")


if __name__ == '__main__':
    # --- Example Usage ---
    data_dir = 'data/pet_expression_classification/'  # Replace with your actual data directory path
    model_name = 'resnet152'  # Replace with your model_name
    analyze_dataset(data_dir, model_name)
