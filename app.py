import os
from flask import Flask, request, render_template, jsonify
from PIL import Image
import torch
from torchvision import transforms
from model import TumorClassifier  # Import the model architecture
import gdown
from tqdm import tqdm

# URL for the model file (use the direct download URL format)
model_url = "https://drive.google.com/uc?id=18R7PokLouAqvucsncurXLBDqTTjs5412"

model_path = "best_model.pth"

# Check if the model file exists, otherwise download it with progress
if not os.path.exists(model_path):
    print("Model file not found. Downloading from Google Drive...")
    gdown.download(model_url, model_path, quiet=False)
    print("Model download complete.")
else:
    print(f"Model file found at {model_path}.")

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
print("Upload folder set up successfully.")

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load model lazily
model = None

def load_model():
    global model
    if model is None:
        print("Loading model...")
        model = TumorClassifier(num_classes=4)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print("Model loaded successfully.")
    else:
        print("Model already loaded.")

# Define image preprocessing pipeline
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
print("Image preprocessing pipeline set up.")

# Map class indices to labels
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
print("Class names defined.")

# Route for homepage
@app.route('/', methods=['GET', 'POST'])
def index():
    print("Received request at '/' endpoint.")
    if request.method == 'POST':
        print("POST request received. Loading the model...")
        load_model()  # Ensure the model is loaded
        print("Checking for uploaded file...")
        if 'file' not in request.files:
            print("No file part in request.")
            return 'No file part'

        file = request.files['file']
        if file.filename == '':
            print("No selected file.")
            return 'No selected file'

        if file:
            print(f"File received: {file.filename}. Saving to disk...")
            # Save the file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            print(f"File saved at {file_path}.")

            print("Performing image preprocessing...")
            # Perform prediction
            image = Image.open(file_path).convert('RGB')
            image_tensor = data_transforms(image).unsqueeze(0).to(device)
            print("Image preprocessed. Running model prediction...")
            with torch.no_grad():
                output = model(image_tensor)
                _, predicted_class = torch.max(output, 1)
            predicted_label = class_names[predicted_class.item()]
            print(f"Prediction complete. Predicted label: {predicted_label}")

            # Pass the result to the template
            print("Rendering template with results.")
            return render_template('index.html', uploaded_image=file_path, result=predicted_label)

    print("Rendering index template for GET request.")
    return render_template('index.html')

if __name__ == "__main__":
    print("Starting Flask app...")
    app.run(debug=True)
