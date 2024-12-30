"""
Script for testing the model with custom images.
"""
import torch
from torchvision import transforms
from PIL import Image
import os
from utils.helpers import load_config, load_model, create_model


def classify_image(image_path, model_path, class_labels, img_size, device):
    """
    Classify a single image using the trained model.

    Args:
        image_path (str): Path to the input image.
        model_path (str): Path to the trained model file.
        class_labels (list): List of class names.
        img_size (int): Size to which the image will be resized.
         device (str)
    Returns:
        str: Predicted class label.
    """
    # Preprocess the image
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error: Unable to open the image {image_path}, check that the image is valid: {e}")
        return None

    image = transform(image).unsqueeze(0).to(device)

    # Load the model
    model = load_model(create_model(num_classes=len(class_labels), model_name=model_name), model_path)
    model.to(device)
    model.eval()

    # Predict
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_labels[predicted.item()]

    return predicted_class


if __name__ == "__main__":
    # Load config
    config_path = 'config.yaml'  # Path to your config
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
    config = load_config(config_path)
    model_name = config['model']
    results_dir = config['results_dir']
    model_results_dir = os.path.join(results_dir, model_name)
    img_size = config['img_size']

    # Example usage
    IMAGE_PATH = "path/to/your/image.jpg"
    MODEL_PATH = os.path.join(model_results_dir, f"{model_name}_best.pth")  # Updated to load the best model
    CLASS_LABELS_PATH = os.path.join(config['data_dir'], 'train')

    if not os.path.exists(CLASS_LABELS_PATH):
        print(f"Error: Class labels not found at {CLASS_LABELS_PATH}")
    else:
        CLASS_LABELS = os.listdir(CLASS_LABELS_PATH)
        if not os.path.exists(MODEL_PATH):
            print(f"Error: Model not found at {MODEL_PATH}")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            prediction = classify_image(IMAGE_PATH, MODEL_PATH, CLASS_LABELS, img_size, device)
            if prediction:
                print(f"The predicted class is: {prediction}")
