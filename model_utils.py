import torch
import timm
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io

# 1. Define the EXACT same architecture as your training
class AIDetectorViT(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224'):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=False)
        n_features = self.model.head.in_features
        self.model.head = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.model(x)

# 2. Initialization
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AIDetectorViT()
model.load_state_dict(torch.load('deepdetect_final.pth', map_location=device))
model.to(device)
model.eval()

# 3. Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        probability = torch.sigmoid(model(tensor)).item()
    
    # Tiered Logic
    if probability > 0.85:
        label = "AI GENERATED or EDITED"
        status = "High Confidence"
    elif probability > 0.6:
        label = "POTENTIALLY AI GENERATED"
        status = "Low Confidence / Artifacts Detected"
    else:
        label = "REAL IMAGE"
        status = "Verified Authentic"
        
    confidence = probability if probability > 0.5 else 1 - probability
    return label, round(confidence * 100, 2), status