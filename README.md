# Handwritten Digit Recognition with PyTorch
This repository contains code for training a deep learning model to recognize handwritten digits using PyTorch. The model is trained on the **MNIST dataset** and tested on custom images. The datasets are downloaded on the process of training in "DL_final_project.ipynb". 

## Features

Improved data loading with data augmentation techniques.
Enhanced model architecture for better performance.
Training loop with learning rate scheduling for improved convergence.

## Requirements
- Python 3.11
- PyTorch
- torchvision
- matplotlib
- Pillow

## usage: to predict for input image
1. Clone the repository:
``` bash
git clone https://github.com/your-username/DL-final-project.git
``` 
3. navigate to the folder
``` bash
cd DL-final-project
```
5. install necessary libraries
``` bash
pip install -r requirements.txt
```
7. run predict.py
``` bash
python predict.py
```
9. Insert image path
10. It will use pre trained parameters saved as "saved_parameters.pth" to predict for the image.


## Model Architecture
The convolutional neural network (CNN) model architecture consists of two convolutional layers followed by max pooling and dropout layers, and two fully connected layers. The architecture has been fine-tuned for improved performance on the handwritten digit recognition task.

## Training and Testing
The model is trained using the MNIST dataset, with data augmentation applied during training. After training, the model is tested on a separate test set to evaluate its accuracy and performance.

## Results
The trained model achieves high accuracy in recognizing handwritten digits, demonstrating its effectiveness in the digit recognition task.
