"""
Computes metrics like precision, recall, F1-score, and confusion matrix.
"""
import os
import json
import torch
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

def compute_metrics(y_true, y_pred, class_labels):
    report = classification_report(y_true, y_pred, target_names=class_labels, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    return report, cm, precision, recall, f1

def plot_confusion_matrix(cm, class_labels, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    plt.close()

def evaluate_model(model, data_loader, device, class_labels, results_dir):
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    report, cm, precision, recall, f1 = compute_metrics(all_labels, all_preds, class_labels)
    
    # Compute accuracy
    total_correct = sum(p == l for p, l in zip(all_preds, all_labels))
    accuracy = total_correct / len(all_labels) * 100

    print(f"Accuracy: {accuracy:.2f}%, "
          f"Precision: {precision:.4f}, "
          f"Recall: {recall:.4f}, F1: {f1:.4f}, "
          f"Confusion Matrix:\n {cm}")

    os.makedirs(results_dir, exist_ok=True)

    # Save metrics
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'classification_report': report
    }
    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    plot_confusion_matrix(cm, class_labels, os.path.join(results_dir, 'confusion_matrix.png'))

    return accuracy, report, cm, precision, recall, f1
