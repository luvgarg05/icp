from flask import Flask, request, jsonify 
from PIL import Image
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
import io
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# Define your image model architecture
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * 56 * 56, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.sm  = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sm(x)
        return x

# Load the image model
model = CNN()
model.load_state_dict(torch.load("Model.pt", map_location=torch.device("cpu")))
model.eval()

# Define image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load Hugging Face conversational model
chat_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
chat_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

# Create Flask app
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted_class = torch.max(outputs, 1)

    return jsonify({"prediction": predicted_class.item()}), 200

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({"error": "No message provided"}), 400

    input_text = data['message']
    chat_history = data.get('history', [])

    # Reconstruct chat history
    new_user_input_ids = chat_tokenizer.encode(input_text + chat_tokenizer.eos_token, return_tensors="pt")
    bot_input_ids = new_user_input_ids

    for past_message in chat_history:
        encoded = chat_tokenizer.encode(past_message + chat_tokenizer.eos_token, return_tensors="pt")
        bot_input_ids = torch.cat([bot_input_ids, encoded], dim=-1)

    chat_history_ids = chat_model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=chat_tokenizer.eos_token_id
    )

    response_text = chat_tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    full_history = chat_history + [input_text, response_text]

    return jsonify({
        "response": response_text,
        "history": full_history
    }), 200

if __name__ == "__main__":
    app.run(debug=True)
