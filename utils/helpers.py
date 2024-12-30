import torch
import yaml
import copy

def load_config(config_path='config/default.yaml'):
    """Loads configuration from a YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_model(model_state, model_path):
    """Saves the model state dictionary to a file."""
    torch.save(model_state, model_path)
    print(f"Best model saved at {model_path}")

def load_model(model, model_path):
    """Loads the model state dictionary from a file."""
    model.load_state_dict(torch.load(model_path, weights_only=True))
    print(f"Best model loaded from {model_path}")
    return model

def create_model(model_name, num_classes):
    """Creates a model based on the model_name"""
    if model_name == "vgg16":
        from models.vgg16 import create_model
    elif model_name == "resnet152":
        from models.resnet152 import create_model
    elif model_name == "densenet121":
        from models.densenet121 import create_model
    elif model_name == "inception_v3":
        from models.inception_v3 import create_model
    elif model_name == "mobilenet":
        from models.mobilenet import create_model
    elif model_name == "pretrained":
        from models.pretrained import create_model
    else:
        raise ValueError(f"Model {model_name} not supported")

    return create_model(num_classes=num_classes)

def early_stopping(val_acc, best_val_acc, epochs_no_improve, patience, model, model_state, precision, recall, f1, best_precision, best_recall, best_f1):
    """Handles early stopping logic."""
    best_model_state = model_state

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = copy.deepcopy(model.state_dict())
        epochs_no_improve = 0
        best_precision = precision
        best_recall = recall
        best_f1 = f1
        print(f"New best model saved with val_acc: {val_acc:.2f}%")
    elif val_acc == best_val_acc:
        if precision > best_precision or recall > best_recall or f1 > best_f1:
            # Update only if precision OR recall OR f1 score improves. It's ok if not all improve as we want the flexibility
            best_precision = precision
            best_recall = recall
            best_f1 = f1
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            print(f"New best model saved with same val_acc but better metrics: Precision:{best_precision:.4f} Recall:{best_recall:.4f} F1:{best_f1:.4f}")
        else:
            epochs_no_improve += 1
    else:
        epochs_no_improve += 1

    return best_val_acc, epochs_no_improve, best_model_state, best_precision, best_recall, best_f1
