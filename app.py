import os
from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import torch
from torchvision import transforms
from model import TumorClassifier  # Import the model architecture

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained model
model = TumorClassifier(num_classes=4)
model.load_state_dict(torch.load('best_model.pth', map_location=device, weights_only=True))
model.to(device)
model.eval()

# Define image preprocessing pipeline
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Map class indices to labels
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']


# Route for homepage
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file is uploaded
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            # Save the file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Perform prediction
            image = Image.open(file_path).convert('RGB')
            image_tensor = data_transforms(image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(image_tensor)
                _, predicted_class = torch.max(output, 1)
            predicted_label = class_names[predicted_class.item()]

            # Pass the result to the template
            return render_template('index.html', uploaded_image=file_path, result=predicted_label)

    return render_template('index.html')


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
