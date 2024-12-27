import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
from utils.data_loader import load_data
from utils.metrics import compute_metrics, plot_confusion_matrix, evaluate_model
from utils.visualization import save_training_plots

def main():
    parser = argparse.ArgumentParser(description="Train a model on pets facial expressions.")
    parser.add_argument("--model", type=str, default="vgg16", help="Name of the model to use (vgg19, resnet152, densenet121, inception_v3, mobilenet, pretrained)")
    parser.add_argument("--data_dir", type=str, default="data/pet_expression_classification/", help="Path to dataset directory")
    parser.add_argument("--results_dir", type=str, default="results/", help="Base directory to save results")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
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
        args.data_dir, img_size=tuple(img_size), batch_size=args.batch_size
    )

    # Create model

    model = create_model(num_classes=len(class_labels))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    history = {
        'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 
        'precision': [], 'recall': [], 'f1': []
    }
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

        print(f"Epoch [{epoch+1}/{args.epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Precision: {precision:.4f}, "
              f"Recall: {recall:.4f}, F1: {f1:.4f}")

    save_training_plots(history, RESULTS_DIR)

    # Save model
    model_path = os.path.join(RESULTS_DIR, f"{args.model}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved at {model_path}")

    # Evaluate model
    print("Validation evaluation running.")
    evaluate_model(model, val_loader, device, class_labels, os.path.join(RESULTS_DIR, 'val'))
    print("Validation evaluation completed. Metrics saved.")

    print("Test evaluation running.")
    evaluate_model(model, test_loader, device, class_labels, os.path.join(RESULTS_DIR, 'test'))
    print("Test evaluation completed. Metrics saved.")

if __name__ == "__main__":
    main()

