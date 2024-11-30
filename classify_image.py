import os
import torch
from torchvision import transforms
from PIL import Image
from model import TumorClassifier  # Import the model architecture

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the saved model
model = TumorClassifier(num_classes=4)
model.load_state_dict(torch.load('best_model.pth', map_location=device, weights_only=True))
model.to(device)
model.eval()  # Set model to evaluation mode

# Define the image preprocessing pipeline
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image
    transforms.ToTensor(),          # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Directory containing images
data_folder = 'data/'

# Map class indices to labels
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Process each image in the folder
for image_name in os.listdir(data_folder):
    image_path = os.path.join(data_folder, image_name)

    try:
        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')  # Ensure image has 3 channels
        image_tensor = data_transforms(image).unsqueeze(0).to(device)  # Add batch dimension

        # Perform inference
        with torch.no_grad():
            output = model(image_tensor)  # Get model predictions
            _, predicted_class = torch.max(output, 1)  # Get the index of the highest score

        # Get the class label
        predicted_label = class_names[predicted_class.item()]

        # Print the result
        print(f'Image: {image_name} - Predicted Class: {predicted_label}')
    except Exception as e:
        print(f'Error processing {image_name}: {e}')
