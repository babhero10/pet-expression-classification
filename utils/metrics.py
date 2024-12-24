"""
Computes metrics like precision, recall, F1-score, and confusion matrix.
"""
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

def compute_metrics(y_true, y_pred, class_labels):
    report = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
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

    # Save metrics
    with open(os.path.join(results_dir, 'classification_report.json'), 'w') as f:
        json.dump(report, f, indent=4)

    plot_confusion_matrix(cm, class_labels, os.path.join(results_dir, 'confusion_matrix.png'))

    return report, cm, precision, recall, f1
