import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from utils.data_loader import load_data
from utils.metrics import compute_metrics, plot_confusion_matrix
from utils.visualization import save_training_plots

def create_model(num_classes=1000):
    model = models.resnet50(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(class_labels))

    # Freeze layers except the last one
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    return model
