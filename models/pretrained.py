import torch.nn as nn
import torchvision.models as models
from torchvision.models import Inception_V3_Weights

def create_model(num_classes):
    """Creates a Inception V3 model with added hidden fully connected layers.

    Args:
        num_classes (int): Number of output classes.
    """

    # Load the pre-trained DenseNet121 model with weights
    weights = Inception_V3_Weights.IMAGENET1K_V1

    model = models.inception_v3(weights=weights)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Modify the final classifier
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),  # Added ReLU activation
        nn.Linear(512, 256),
        nn.ReLU(),  # Added ReLU activation
        nn.Linear(256, num_classes)
    )

    aux_num_ftrs = model.AuxLogits.fc.in_features
    model.AuxLogits.fc = nn.Sequential(
        nn.Linear(aux_num_ftrs, 512),
        nn.ReLU(),  # Added ReLU activation
        nn.Linear(512, 256),
        nn.ReLU(),  # Added ReLU activation
        nn.Linear(256, num_classes)
    )

    return model
