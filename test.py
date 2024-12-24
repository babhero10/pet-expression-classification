"""
Script for testing the model with custom images.
"""
import os
import torch
from torchvision import transforms
from PIL import Image
from models.vgg import create_model  # Replace with appropriate model

def classify_image(image_path, model_path, class_labels, img_size=(224, 224)):
    """
    Classify a single image using the trained model.

    Args:
        image_path (str): Path to the input image.
        model_path (str): Path to the trained model file.
        class_labels (list): List of class names.
        img_size (tuple): Size to which the image will be resized.

    Returns:
        str: Predicted class label.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = create_model(num_classes=len(class_labels))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Preprocess the image
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(image)
        _, predicted = outputs.max(1)
        predicted_class = class_labels[predicted.item()]

    return predicted_class

if __name__ == "__main__":
    # Example usage
    IMAGE_PATH = "path/to/your/image.jpg"
    MODEL_PATH = "results/vgg16/model.pth"
    CLASS_LABELS = ["Angry", "Other", "Sad", "Happy"]  # Replace with your actual class labels

    prediction = classify_image(IMAGE_PATH, MODEL_PATH, CLASS_LABELS)
    print(f"The predicted class is: {prediction}")

