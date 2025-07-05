import cv2

# Variáveis globais
drawing = False
ix, iy = -1, -1
rectangle = (0, 0, 0, 0)


# Função de callback do mouse
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, rectangle, frame_copy

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            frame_copy = frame.copy()
            cv2.rectangle(frame_copy, (ix, iy), (x, y), (0, 255, 0), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        rectangle = (ix, iy, x - ix, y - iy)
        print(f"Retângulo: x={rectangle[0]}, y={rectangle[1]}, largura={rectangle[2]}, altura={rectangle[3]}")
        cv2.rectangle(frame_copy, (ix, iy), (x, y), (0, 255, 0), 2)

# Carrega o vídeo
cap = cv2.VideoCapture("videos/esteiraML.mp4")
ret, frame = cap.read()

if not ret:
    print("Erro ao carregar o vídeo.")
    cap.release()
    exit()

frame_copy = frame.copy()
cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", draw_rectangle)

while True:
    cv2.imshow("Frame", frame_copy)
    key = cv2.waitKey(1) & 0xFF

    # Pressione ESC ou Enter para sair
    if key == 27 or key == 13:
        break

cap.release()
cv2.destroyAllWindows()
