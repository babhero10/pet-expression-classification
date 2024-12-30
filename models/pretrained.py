import torch.nn as nn
import torchvision.models as models
from torchvision.models import DenseNet121_Weights

def create_model(num_classes):
    """Creates a DenseNet121 model with added hidden fully connected layers
    and a modified final layer for a custom number of classes.

    Args:
        num_classes (int): Number of output classes.
    """

    # Load the pre-trained DenseNet121 model with weights
    weights = DenseNet121_Weights.IMAGENET1K_V1
    model = models.densenet121(weights=weights)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.classifier.in_features

    hidden_layers = [512, 256, 128]
    layers = []
    prev_size = num_ftrs

    for hidden_size in hidden_layers:
        layers.append(nn.Linear(prev_size, hidden_size))
        layers.append(nn.ReLU())
        # layers.append(nn.Dropout(0.0))
        prev_size = hidden_size

    layers.append(nn.Linear(prev_size, num_classes))

    model.classifier = nn.Sequential(*layers)

    return model
