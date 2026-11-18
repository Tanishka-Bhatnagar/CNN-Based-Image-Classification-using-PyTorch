import spaces
import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

class AnimalCNN(nn.Module):
    def __init__(self):
        super(AnimalCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

model = AnimalCNN()
model.load_state_dict(torch.load("animal_cnn_weights.pth", map_location="cpu"))
model.eval()

classes = ['Cat', 'Dog', 'Wild']

@spaces.GPU
def predict_animal(input_image):
    if input_image is None:
        return None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    image = input_image.convert("RGB").resize((128, 128))
    
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)
    
    return {classes[i]: float(probabilities[i]) for i in range(3)}

# Gradio App launch
interface = gr.Interface(
    fn=predict_animal,
    inputs=gr.Image(type="pil", label="Upload an Animal Face Photo"),
    outputs=gr.Label(num_top_classes=3, label="Predicted Class & Confidence"),
    title="🐾 Animal Face Classifier (PyTorch CNN)",
    description="Upload a clear photo of a Cat, Dog, or Wild Animal (tiger, lion, wolf, fox). Built with PyTorch and trained on the AFHQ dataset (96.8% validation accuracy)."
)

interface.launch()
