import matplotlib.pyplot as plt
import os

def save_training_plots(history, results_dir):
    """Saves training plots."""
    # Plot Accuracy
    plt.figure()
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.savefig(os.path.join(results_dir, 'accuracy_plot.png'))
    plt.close()

    # Plot Loss
    plt.figure()
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(os.path.join(results_dir, 'loss_plot.png'))
    plt.close()

    # Plot Precision, Recall, F1-Score
    plt.figure()
    plt.plot(history['precision'], label='Precision')
    plt.plot(history['recall'], label='Recall')
    plt.plot(history['f1'], label='F1-Score')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.legend()
    plt.title('Precision, Recall, and F1-Score')
    plt.savefig(os.path.join(results_dir, 'metrics_plot.png'))
    plt.close()
