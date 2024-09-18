"""

@author:  AIcandy 
@website: aicandy.vn

"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from aicandy_model_src_eboxesox.aicandy_mobilenet_model_mhgmyhay import CustomMobileNet


# python aicandy_mobilenet_test_vtvlmtxo.py --image_path ../image_test.jpg --model_path aicandy_model_out_tdtagoyx/aicandy_model_pth_bmdmrcav.pth --label_path label.txt

def load_labels(label_path):
    with open(label_path, 'r') as f:
        labels = {int(line.split(": ")[0]): line.split(": ")[1].strip() for line in f}
    print('labels: ',labels)
    return labels

def predict(image_path, model_path, label_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    labels = load_labels(label_path)
    num_classes = len(labels)
    
    model = CustomMobileNet(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image)
        _, predicted = outputs.max(1)
    
    predicted_label = labels[predicted.item()]
    print(f'Predicted Label: {predicted_label}')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='AIcandy.vn')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the image to be predicted')
    parser.add_argument('--model_path', type=str, default='best_model.pth', help='Path to the trained model')
    parser.add_argument('--label_path', type=str, default='label.txt', help='Path to the label file')
    
    args = parser.parse_args()
    
    predict(image_path=args.image_path, model_path=args.model_path, label_path=args.label_path)
