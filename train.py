import os
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from utils.data_loader import load_data
from utils.metrics import compute_metrics
from utils.visualization import save_training_plots
from utils.helpers import load_config, save_model, create_model, early_stopping
import yaml


def train_step(model, data_loader, criterion, optimizer, device, accumulation_steps=1,
               aux_loss_weight=0.4):
    """Performs a single training step with optional gradient accumulation."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    optimizer.zero_grad()

    for images, labels in tqdm.tqdm(data_loader, desc='Training loop'):  # Corrected Line
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)

        if isinstance(outputs, tuple):
            main_out, aux_out = outputs
            main_loss = criterion(main_out, labels)
            aux_loss = criterion(aux_out, labels)
            loss = main_loss + aux_loss * aux_loss_weight
        else:
            loss = criterion(outputs, labels)

        # Scale the loss by the accumulation steps
        loss = loss / accumulation_steps
        loss.backward()

        # Perform optimizer step every `accumulation_steps`
        i = len(all_labels) // data_loader.batch_size
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(data_loader):
            optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item() * accumulation_steps

        if isinstance(outputs, tuple):
            main_out, _ = outputs
            _, predicted = torch.max(main_out, 1)
        else:
            _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu())
        all_labels.extend(labels.cpu())

    avg_loss = running_loss / len(data_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy, all_labels, all_preds


def val_step(model, data_loader, criterion, device, aux_loss_weight=0.4):
    """Performs a single validation step."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:  # Corrected Line

            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            if isinstance(outputs, tuple):
                main_out, aux_out = outputs
                main_loss = criterion(main_out, labels)
                aux_loss = criterion(aux_out, labels)
                loss = main_loss + aux_loss * aux_loss_weight
            else:
                loss = criterion(outputs, labels)

            running_loss += loss.item()

            if isinstance(outputs, tuple):
                main_out, _ = outputs
                _, predicted = torch.max(main_out, 1)
            else:
                _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu())
            all_labels.extend(labels.cpu())

    avg_loss = running_loss / len(data_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy, all_labels, all_preds


def main():
    config = load_config()

    # Set a manual seed for reproducibility
    seed = config['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # Enable TF32 for better performance
    torch.set_float32_matmul_precision('high')

    # Set up directories
    model_name = config['model']
    results_dir = config['results_dir']
    model_results_dir = os.path.join(results_dir, model_name)
    os.makedirs(model_results_dir, exist_ok=True)

    # Load data
    train_loader, val_loader, _, class_labels = load_data(
        config['data_dir'],
        batch_size=config['batch_size'],
        model_name=model_name,
        seed=seed
    )

    print(f"Data loaded with {len(class_labels)} classes: {class_labels}")
    print(f"Train size: {len(train_loader.dataset)}, Val size: {len(val_loader.dataset)}")

    # Create model
    model = create_model(model_name, num_classes=len(class_labels))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=float(config['learning_rate']), weight_decay=float(config['weight_decay']))
    scheduler = StepLR(optimizer, step_size=config['step_size'], gamma=config['gamma'])

    # Training loop
    history = {
        'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [],
        'precision': [], 'recall': [], 'f1': []
    }

    aux_loss_weight = float(config['aux_loss_weight'])
    best_val_acc = 0.0
    best_model_state = None
    best_precision = 0.0
    best_recall = 0.0
    best_f1 = 0.0
    epochs_no_improve = 0

    for epoch in range(config['epochs']):
        train_loss, train_acc, _, _ = train_step(
            model, train_loader, criterion, optimizer, device,
            accumulation_steps=config['accumulation_steps'],
            aux_loss_weight=aux_loss_weight
        )

        val_loss, val_acc, all_labels, all_preds = val_step(
            model, val_loader, criterion, device, aux_loss_weight=aux_loss_weight
        )

        report, cm, precision, recall, f1 = compute_metrics(torch.tensor(all_labels), torch.tensor(all_preds), class_labels)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['precision'].append(precision)
        history['recall'].append(recall)
        history['f1'].append(f1)

        print(f"Epoch [{epoch + 1}/{config['epochs']}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Precision: {precision:.4f}, "
              f"Recall: {recall:.4f}, F1: {f1:.4f}, "
              f"Learning rate: {optimizer.param_groups[0]['lr']}")

        # Early stopping
        best_val_acc, epochs_no_improve, best_model_state, best_precision, best_recall, best_f1 = early_stopping(
            val_acc, best_val_acc, epochs_no_improve, config['patience'], model, best_model_state, precision, recall, f1, best_precision, best_recall, best_f1
        )

        if epochs_no_improve >= config['patience'] and config['early_stopping']:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

        scheduler.step()

    save_training_plots(history, model_results_dir)
    if best_model_state:
        model_path = os.path.join(model_results_dir, f"{model_name}_best.pth")
        save_model(best_model_state, model_path)
    else:
        print("No best model was saved, training did not improve.")

    # Save the args to a YAML file
    with open(os.path.join(model_results_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

if __name__ == "__main__":
    main()
