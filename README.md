# Pets expression classification
This project is a simple example of how to classify pets expressions using a Convolutional Neural Network (CNN) with different models like 
**VGG16**, **ResNet152**, **InceptionV3**, **MobileNet** and **Densenet121**

## Dataset
The dataset used in this project is the [Pets expressions dataset](https://www.kaggle.com/datasets/anshtanwar/pets-facial-expression-dataset) from Kaggle. This dataset contains 1000 images of pets expressions with 4 different classes: Angry, Happy, Sad and Other.

## Usage
To download the dataset, you can use the following command:
```
python -m download
```

To train the model, you can use the following command:
```
python -m train
```

To test the model (Not working yet), you can use the following command:
```
python -m test
```

Configs:
- `--model`: Model to use (vgg16, resnet152, inceptionv3, mobilenet, densenet121).
- `--epochs`: Number of epochs to train the model.
- `--lr`: Learning rate.
- `--batch_size`: Batch size.
- `--wegith_decay`: Weight decay.

