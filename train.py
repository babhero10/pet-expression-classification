import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import os
import argparse
import yaml
from utils.data_loader import load_data
from utils.metrics import compute_metrics, evaluate_model
from utils.visualization import save_training_plots
import copy

def main():
    parser = argparse.ArgumentParser(description="Train a model on pets facial expressions.")
    parser.add_argument("--model", type=str, default="vgg16", help="Name of the model to use (vgg19, resnet152, densenet121, inception_v3, mobilenet, pretrained)")
    parser.add_argument("--data_dir", type=str, default="data/pet_expression_classification/", help="Path to dataset directory")
    parser.add_argument("--results_dir", type=str, default="results/", help="Base directory to save results")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-1, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--step_size", type=int, default=5, help="Step size for learning rate scheduler")
    parser.add_argument("--gamma", type=float, default=0.1, help="Gamma for learning rate scheduler")
    args = parser.parse_args()

    RESULTS_DIR = os.path.join(args.results_dir, args.model)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    img_size = (224, 224)

    if args.model == "vgg19":
        from models.vgg19 import create_model
    elif args.model == "resnet152":
        from models.resnet152 import create_model
    elif args.model == "densenet121":
        from models.densenet121 import create_model
    elif args.model == "inception_v3":
        from models.inception_v3 import create_model
        img_size = (299, 299)
    elif args.model == "mobilenet":
        from models.mobilenet import create_model
    elif args.model == "pretrained":
        from models.pretrained import create_model
    else:
        raise ValueError(f"Model {args.model} not supported")

    # Load data
    train_loader, val_loader, test_loader, class_labels = load_data(
        args.data_dir, img_size=tuple(img_size), batch_size=args.batch_size, num_augmentations=0
    )

    print(f"Data loaded with {len(class_labels)} classes: {class_labels}")
    print(f"Train size: {len(train_loader.dataset)}, Val size: {len(val_loader.dataset)}, Test size: {len(test_loader.dataset)}")

    # Create model
    model = create_model(num_classes=len(class_labels))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Define StepLR scheduler
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # Training loop
    history = {
        'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [],
        'precision': [], 'recall': [], 'f1': []
    }

    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = 100. * correct / total

        report, cm, precision, recall, f1 = compute_metrics(all_labels, all_preds, class_labels)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['precision'].append(precision)
        history['recall'].append(recall)
        history['f1'].append(f1)

        print(f"Epoch [{epoch + 1}/{args.epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Precision: {precision:.4f}, "
              f"Recall: {recall:.4f}, F1: {f1:.4f}, "
              f"Learning rate: {optimizer.param_groups[0]['lr']}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"Epoch {epoch + 1}: New best model saved with val_acc: {val_acc:.2f}%")

        # Step the scheduler after each epoch
        scheduler.step()

    save_training_plots(history, RESULTS_DIR)

    # Save the best model
    if best_model_state:
        model_path = os.path.join(RESULTS_DIR, f"{args.model}_best.pth")
        torch.save(best_model_state, model_path)
        print(f"Best model saved at {model_path}")
    else:
        print("No best model was saved, training did not improve.")

    # Save the args to a YAML file
    with open(os.path.join(RESULTS_DIR, 'args.yaml'), 'w') as f:
        yaml.dump(vars(args), f)

    # Load the best model for evaluation
    model = create_model(num_classes=len(class_labels)).to(device)

    # Load the best model for evaluation
    model.load_state_dict(best_model_state)
    print("Best model loaded for evaluation")

    # Evaluate model
    print("Training evaluation running.")
    evaluate_model(model, train_loader, device, class_labels, os.path.join(RESULTS_DIR, 'train'))
    print("Training evaluation completed. Metrics saved.")

    print("Validation evaluation running.")
    evaluate_model(model, val_loader, device, class_labels, os.path.join(RESULTS_DIR, 'val'))
    print("Validation evaluation completed. Metrics saved.")

    print("Test evaluation running.")
    evaluate_model(model, test_loader, device, class_labels, os.path.join(RESULTS_DIR, 'test'))
    print("Test evaluation completed. Metrics saved.")

if __name__ == "__main__":
    main()
