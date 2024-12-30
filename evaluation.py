import torch
import os
from utils.data_loader import load_data
from utils.metrics import evaluate_model
from utils.helpers import load_config, load_model, create_model

def combine_datasets(loaders):
    """Combines multiple dataloaders into a single dataset."""
    combined_dataset = torch.utils.data.ConcatDataset([loader.dataset for loader in loaders])
    return torch.utils.data.DataLoader(combined_dataset, batch_size=loaders[0].batch_size, shuffle=False, pin_memory=True, num_workers=4)


def main():
    config = load_config()

    # Set up directories
    model_name = config['model']
    results_dir = config['results_dir']
    model_results_dir = os.path.join(results_dir, model_name)
    os.makedirs(model_results_dir, exist_ok=True)

    # Load Data
    train_loader, val_loader, test_loader, class_labels = load_data(
        config['data_dir'],
        batch_size=config['batch_size'],
        model_name=model_name
    )

    # Combine datasets for overall evaluation
    combined_loader = combine_datasets([train_loader, val_loader, test_loader])

    print(f"Data loaded with {len(class_labels)} classes: {class_labels}")
    print(f"Train size: {len(train_loader.dataset)}, Val size: {len(val_loader.dataset)}, Test size: {len(test_loader.dataset)}")
    print(f"Combined size: {len(combined_loader.dataset)}")

    # Create model
    model = create_model(model_name, num_classes=len(class_labels))

    # Load model
    model_path = os.path.join(model_results_dir, f"{model_name}_best.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Best model not found at {model_path}")
    model = load_model(model, model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Evaluate on individual datasets
    print("Training evaluation running.")
    evaluate_model(model, train_loader, device, class_labels, os.path.join(model_results_dir, 'train'))
    print("Training evaluation completed. Metrics saved.")

    print("Validation evaluation running.")
    evaluate_model(model, val_loader, device, class_labels, os.path.join(model_results_dir, 'val'))
    print("Validation evaluation completed. Metrics saved.")

    print("Test evaluation running.")
    evaluate_model(model, test_loader, device, class_labels, os.path.join(model_results_dir, 'test'))
    print("Test evaluation completed. Metrics saved.")

    # Evaluate on combined dataset
    print("Overall evaluation running.")
    evaluate_model(model, combined_loader, device, class_labels, os.path.join(model_results_dir, 'overall'))
    print("Overall evaluation completed. Metrics saved.")

if __name__ == "__main__":
    main()
