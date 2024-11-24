import sys
import os

# Adicionando o diretório raiz do projeto (o diretório onde 'app' e 'models' estão localizados)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Ajuste conforme necessário

# Agora você pode importar o GradCAM da pasta 'app.utils'
from app.utils.gradcam import GradCAM
from torchvision import transforms
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# Carregar o modelo YOLOv8 para classificação
MODEL = YOLO("models\\best_binary.pt")
# Selecionar a camada-alvo
TARGET_LAYER = MODEL.model.model[9].conv

def aply_gradcam_binary(image_path:str) -> str:
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0)

    # Aplicar o Grad-CAM
    grad_cam = GradCAM(MODEL, TARGET_LAYER)
    cam, predicted_class, confidence = grad_cam(input_tensor)

    # Mapeamento da classe predita
    class_names = {0: 'benign', 1: 'malignant'}
    predicted_label = class_names[predicted_class]
    print(f"Classe predita: {predicted_label} com {confidence:.2f} de certeza.")

    # Sobrepor Grad-CAM na imagem original
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    image_np = np.array(image)
    heatmap = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))
    superimposed_img = cv2.addWeighted(image_np, 0.6, heatmap, 0.4, 0.0)  # Adicionado gamma=0.0

    # Salvar e exibir a imagem com Grad-CAM
    output_path = f"outputs\\{(image_path.split("\\")[1]).split(".")[0]}_gradcam.png"
    cv2.imwrite(output_path, superimposed_img)
    print(f"Grad-CAM result saved at {output_path}")
    
    return predicted_label