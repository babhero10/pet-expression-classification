"""
Script for applying transfer learning using pre-trained ImageNet weights.
"""
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from utils.data_loader import load_data
from utils.metrics import compute_metrics, plot_confusion_matrix, evaluate_model
from utils.visualization import save_training_plots

DATA_DIR = "data/pets_facial_expressions/"
RESULTS_DIR = "results/transfer_learning/"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5

os.makedirs(RESULTS_DIR, exist_ok=True)

# Load data
train_loader, val_loader, test_loader, class_labels = load_data(DATA_DIR, img_size=IMG_SIZE, batch_size=BATCH_SIZE)

# Load pre-trained model
model = models.resnet50(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(class_labels))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Freeze layers except the last one
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.0001)

# Training loop
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'precision': [], 'recall': [], 'f1': []}
for epoch in range(EPOCHS):
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

    print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

save_training_plots(history, RESULTS_DIR)

# Save model
torch.save(model.state_dict(), os.path.join(RESULTS_DIR, 'model.pth'))

evaluate_model(model, val_loader, device, class_labels, RESULTS_DIR+'/val')
print("Validation evaluation completed. Metrics saved.")

evaluate_model(model, test_loader, device, class_labels, RESULTS_DIR+'/test')
print("Test evaluation completed. Metrics saved.")
