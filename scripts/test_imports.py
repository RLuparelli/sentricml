#!/usr/bin/env python3
"""
Script de teste das importações
"""

import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO

print("SUCESSO: Todas as importacoes funcionaram!")
print(f"OpenCV: {cv2.__version__}")
print(f"NumPy: {np.__version__}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA disponivel: {torch.cuda.is_available()}")

# Teste basico
img = np.zeros((100, 100, 3), dtype=np.uint8)
cv2.rectangle(img, (10, 10), (90, 90), (255, 255, 255), 2)
pil_img = Image.fromarray(img)
tensor = torch.tensor([1.0, 2.0, 3.0])

print("SUCESSO: Testes basicos passaram!")
print("Sistema pronto para uso!")
