from flask import Flask, render_template, request, redirect, url_for
import torch
from torchvision import transforms, datasets
from PIL import Image
import os
from werkzeug.utils import secure_filename

# Import your model + training setup
from scr.Plant_Vilage.components.model_trainer import get_model_optimizer

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model, optimizer, lossFun = get_model_optimizer()
MODEL_PATH = "artifacts/model_trainer/cnn_model.pth"
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Preprocessing (must match what you used during training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # or whatever you used
        std=[0.229, 0.224, 0.225]
    )
])

# Load the dataset purely to get class names (these are folder names in your train set)
dataset_for_classes = datasets.ImageFolder(
    root="artifacts/data_injection/dataset/Plant_Vilage_dataset/train_test_data/PLATE_train",
    transform=transform
)
class_names = dataset_for_classes.classes  # List of strings, length ~39

app = Flask(__name__)
os.makedirs("static", exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/train", methods=["GET", "POST"])
def train():
    if request.method == "POST":
        # Run your training script
        os.system("python main.py")
        return render_template("train.html", message="✅ Model trained and saved!")
    return render_template("train.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")
    if not file:
        return redirect(url_for("home"))

    # Save the uploaded image
    filename = secure_filename(file.filename)
    file_path = os.path.join("static", filename)
    file.save(file_path)

    # Load & preprocess
    image = Image.open(file_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)  # shape [1, C, H, W]

    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        predicted_idx = outputs.argmax(dim=1).item()

    # Map to class name
    if predicted_idx < len(class_names):
        label = class_names[predicted_idx]
    else:
        label = f"Unknown (idx={predicted_idx})"

    # Also pass probabilities + class_names to template
    # Optionally only show top few probabilities
    probs_list = probs.cpu().numpy().tolist()

    return render_template("result.html",
                           image=file_path,
                           label=label,
                           classes=class_names,
                           probs=probs_list)

if __name__ == "__main__":
    app.run(debug=True)
