import torch
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt

# Load the saved model
from model import CNN  # Assuming your model architecture is saved in model.py

# Load the saved model parameters
model = CNN()
model.load_state_dict(torch.load('saved_parameters.pth'))
model.eval()

# Preprocess the image
def preprocess_image(image_path):
    image = Image.open(image_path).convert('L')
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Invert pixel values to have black background and white digits
    inverted_image_tensor = 1 - image_tensor
    return image, inverted_image_tensor

# Function to predict digit
def predict_digit(image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        predicted_label = output.argmax(1).item()
    return predicted_label

def main():
    # User input for image file path
    image_path = input("Enter the path to the image of the digit: ")

    # Preprocess the image
    original_image, preprocessed_image_tensor = preprocess_image(image_path)

    # Predict digit
    predicted_digit = predict_digit(preprocessed_image_tensor)

    # Display original image, preprocessed image, and prediction
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    preprocessed_image = preprocessed_image_tensor.squeeze(0).permute(1, 2, 0).numpy()
    axes[1].imshow(preprocessed_image, cmap='gray')
    axes[1].set_title(f"Preprocessed Image\nPredicted Digit: {predicted_digit}")
    axes[1].axis('off')

    plt.show()

if __name__ == "__main__":
    main()
