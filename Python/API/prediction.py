from flask import Flask, request, jsonify
from PIL import Image
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import io

# Define your model architecture (same as in training)
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * 56 * 56, 512)  # Adjust input size based on your image size
        self.fc2 = nn.Linear(512, num_classes)
        self.sm  = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sm(x)
        return x
# Load model
model = CNN()
model.load_state_dict(torch.load("Model.pt", map_location=torch.device("cpu")))
model.eval()

# Define image transforms (must match what you used in training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # or whatever size your model expects
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # example normalization
])

# Create Flask app
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']

    # Read image file as bytes
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # Apply transforms
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted_class = torch.max(outputs, 1)

    return jsonify({
        "prediction": predicted_class.item()
    }), 200

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
