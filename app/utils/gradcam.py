import numpy as np
import cv2
import torch
import torch.nn.functional as F

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Registrar hooks para capturar gradientes e ativações
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        """Capturar ativações no forward pass."""
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        """Capturar gradientes no backward pass."""
        self.gradients = grad_output[0]

    def __call__(self, input_tensor):
        # Certificar-se de que a entrada rastreia gradientes
        input_tensor.requires_grad = True

        # Forward pass diretamente no modelo interno
        raw_output = self.model.model(input_tensor)  # Acesso ao modelo interno
        predicted_class = raw_output.argmax(dim=1).item()  # Identificar a classe predita
        class_score = raw_output[0, predicted_class]  # Selecionar o escore da classe predita

        # Calcular a probabilidade da classe predita (caso o modelo não tenha softmax, aplique manualmente)
        probabilities = torch.softmax(raw_output, dim=1)  # Aplica softmax para obter probabilidades
        confidence = probabilities[0, predicted_class].item() * 100  # Multiplica por 100 para percentual

        # Backward pass
        self.model.zero_grad()
        class_score.backward(retain_graph=True)

        # Gradientes e ativações
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]

        # Calcular Grad-CAM
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU para remover valores negativos
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input_tensor.shape[2:])  # Redimensionar para o tamanho da imagem original
        cam = cam / cam.max()  # Normalizar

        return cam, predicted_class, confidence  # Retornar o mapa de calor, a classe predita e a confiança