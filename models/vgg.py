from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models

def create_model(input_shape=(224, 224, 3), num_classes=7):
    base_model = VGG16(weights=None, include_top=False, input_shape=input_shape)
    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model
