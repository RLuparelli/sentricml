import cv2
import os
from ultralytics import YOLO

# Caminhos
modelo_path = r"C:\Users\lupar\Desktop\Sentric\MercadoLivre\Scripts\MercadoLivreBest.pt"
video_input_path = r"C:\Users\lupar\Desktop\Sentric\MercadoLivre\Scripts\videos\1771581.mp4"
output_dir = r"C:\Users\lupar\Desktop\Sentric\MercadoLivre\Scripts\videosProcessados"
video_output_path = os.path.join(output_dir, "1771581_processado.mp4")

# Define a área de contagem (x, y, largura, altura)
roi = (444, 0, 343, 298)

# Carrega o modelo
model = YOLO(modelo_path)

# Abre o vídeo de entrada
cap = cv2.VideoCapture(video_input_path)
if not cap.isOpened():
    raise IOError(f"Erro ao abrir vídeo: {video_input_path}")

# Configura o vídeo de saída
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))

# Loop pelos frames
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inference
    results = model(frame, verbose=False)[0]

    # Desenha a ROI
    x, y, w, h = roi
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    count = 0
    # Para cada detecção
    for box in results.boxes:
        bx, by, bw, bh = box.xyxy[0].tolist()  # Bounding box [x1, y1, x2, y2]
        cx = (bx + bw) / 2
        cy = (by + bh) / 2

        # Verifica se o centro está dentro da ROI
        if x <= cx <= x + w and y <= cy <= y + h:
            count += 1
            cv2.rectangle(frame, (int(bx), int(by)), (int(bw), int(bh)), (0, 255, 0), 2)

    # Exibe a contagem no vídeo
    cv2.putText(frame, f"Pacotes na ROI: {count}", (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Escreve o frame no vídeo de saída
    out.write(frame)

# Libera os recursos
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Vídeo processado salvo em: {video_output_path}")
