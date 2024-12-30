import os
import json
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def compute_metrics(y_true, y_pred, class_labels):
    """Computes metrics using tensors directly."""
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    report = classification_report(y_true, y_pred, target_names=class_labels, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    # Compute weighted precision, recall and f1 from report
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1 = report['weighted avg']['f1-score']

    return report, cm, precision, recall, f1

def plot_confusion_matrix(cm, class_labels, save_path):
    """Plots the confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.2, style='whitegrid')
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    plt.close()


def evaluate_model(model, data_loader, device, class_labels, results_dir):
    """Evaluates the model on a given data loader."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu())
            all_labels.extend(labels.cpu())

    all_preds = torch.tensor(all_preds)
    all_labels = torch.tensor(all_labels)

    report, cm, precision, recall, f1 = compute_metrics(all_labels, all_preds, class_labels)
    accuracy = (all_preds == all_labels).float().mean() * 100

    print(f"Accuracy: {accuracy:.2f}%, "
          f"Precision: {precision:.4f}, "
          f"Recall: {recall:.4f}, F1: {f1:.4f}")
    print(f"Confusion Matrix:\n {cm}")

    os.makedirs(results_dir, exist_ok=True)

    # Save metrics
    metrics = {
        'accuracy': accuracy.item(),
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'classification_report': report
    }
    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    plot_confusion_matrix(cm, class_labels, os.path.join(results_dir, 'confusion_matrix.png'))
