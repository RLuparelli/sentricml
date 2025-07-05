import cv2
from ultralytics import YOLO
import os
import numpy as np

# Configurações
VIDEO_PATH = r"C:\Users\lupar\Desktop\Sentric\MercadoLivre\Scripts\videos\esteiraML.mp4"
MODEL_PATH = r"C:\Users\lupar\Desktop\Sentric\MercadoLivre\Scripts\MercadoLivreBest.pt"
OUTPUT_PATH = r"C:\Users\lupar\Desktop\Sentric\MercadoLivre\Scripts\videos\esteiraML_zoomed_tracked.mp4"

# REGIÃO DE INTERESSE (ROI)
ROI_X = 179
ROI_Y = 386
ROI_WIDTH = 425
ROI_HEIGHT = 334

# CONFIGURAÇÕES DE ZOOM
ZOOM_FACTOR = 2  # Fator de zoom (2.0 = dobra o tamanho)
TARGET_SIZE = 640  # Tamanho alvo para a ROI ampliada (padrão YOLO)

# Tentar importar tracking
try:
    from trackers import SORTTracker
    TRACKING_AVAILABLE = True
    print("✓ Biblioteca de tracking disponível")
except ImportError:
    TRACKING_AVAILABLE = False
    print("⚠ Biblioteca de tracking não disponível")

try:
    import supervision as sv
    SUPERVISION_AVAILABLE = True
    print("✓ Supervision disponível")
except ImportError:
    SUPERVISION_AVAILABLE = False
    print("⚠ Supervision não disponível")

def extract_and_resize_roi(frame, x, y, width, height, target_size=None, zoom_factor=None):
    """
    Extrai ROI e aplica zoom/redimensionamento para melhorar inferência
    """
    frame_h, frame_w = frame.shape[:2]
    
    # Garante que as coordenadas estão dentro dos limites
    x = max(0, min(x, frame_w - 1))
    y = max(0, min(y, frame_h - 1))
    width = min(width, frame_w - x)
    height = min(height, frame_h - y)
    
    # Extrai ROI original
    roi = frame[y:y+height, x:x+width]
    original_roi_shape = roi.shape[:2]
    
    # Determina novo tamanho baseado no método escolhido
    if target_size is not None:
        # Redimensiona mantendo aspect ratio para tamanho alvo
        aspect_ratio = width / height
        if aspect_ratio > 1:  # Mais largo que alto
            new_width = target_size
            new_height = int(target_size / aspect_ratio)
        else:  # Mais alto que largo
            new_height = target_size
            new_width = int(target_size * aspect_ratio)
    elif zoom_factor is not None:
        # Aplica fator de zoom
        new_width = int(width * zoom_factor)
        new_height = int(height * zoom_factor)
    else:
        # Sem redimensionamento
        new_width, new_height = width, height
    
    # Redimensiona ROI se necessário
    if (new_width, new_height) != (width, height):
        # Usa interpolação cúbica para melhor qualidade na ampliação
        roi_resized = cv2.resize(roi, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        scale_factor = (new_width / width, new_height / height)
        print(f"ROI ampliada de {width}x{height} para {new_width}x{new_height} (escala: {scale_factor[0]:.2f}x)")
    else:
        roi_resized = roi
        scale_factor = (1.0, 1.0)
    
    return roi_resized, (x, y, width, height), scale_factor

def convert_detections_to_original_frame(detections, roi_coords, scale_factor):
    """
    Converte coordenadas das detecções da ROI ampliada para o frame original
    """
    roi_x, roi_y, roi_w, roi_h = roi_coords
    scale_x, scale_y = scale_factor
    
    if SUPERVISION_AVAILABLE and len(detections) > 0:
        # Para supervision Detections
        # Primeiro, converte da escala ampliada para a ROI original
        detections.xyxy[:, 0] /= scale_x  # x1
        detections.xyxy[:, 1] /= scale_y  # y1
        detections.xyxy[:, 2] /= scale_x  # x2
        detections.xyxy[:, 3] /= scale_y  # y2
        
        # Depois, converte para coordenadas do frame completo
        detections.xyxy[:, 0] += roi_x  # x1
        detections.xyxy[:, 1] += roi_y  # y1
        detections.xyxy[:, 2] += roi_x  # x2
        detections.xyxy[:, 3] += roi_y  # y2
        
        return detections
    else:
        # Para detecções simples (lista)
        adjusted_detections = []
        for detection in detections:
            x1, y1, x2, y2, conf = detection
            
            # Converte da escala ampliada para ROI original
            x1_orig = x1 / scale_x
            y1_orig = y1 / scale_y
            x2_orig = x2 / scale_x
            y2_orig = y2 / scale_y
            
            # Converte para frame completo
            x1_final = x1_orig + roi_x
            y1_final = y1_orig + roi_y
            x2_final = x2_orig + roi_x
            y2_final = y2_orig + roi_y
            
            adjusted_detections.append([x1_final, y1_final, x2_final, y2_final, conf])
        
        return adjusted_detections

def draw_roi_and_detections(frame, roi_coords, detections, colors, zoom_info=None):
    """Desenha ROI e detecções no frame"""
    roi_x, roi_y, roi_w, roi_h = roi_coords
    
    # Desenha retângulo da ROI
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 255, 255), 2)
    
    # Label da ROI com informação de zoom
    if zoom_info:
        label = f'ROI (Zoom: {zoom_info:.1f}x)'
    else:
        label = 'ROI'
    
    cv2.putText(frame, label, (roi_x, roi_y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Desenha detecções
    detection_count = 0
    
    if SUPERVISION_AVAILABLE and len(detections) > 0:
        for i, bbox in enumerate(detections.xyxy):
            x1, y1, x2, y2 = map(int, bbox)
            
            # Verifica se a detecção está dentro da ROI (sanity check)
            if (x1 >= roi_x and y1 >= roi_y and 
                x2 <= roi_x + roi_w and y2 <= roi_y + roi_h):
                
                # Cor baseada no tracker_id se disponível
                if hasattr(detections, 'tracker_id') and detections.tracker_id is not None:
                    track_id = detections.tracker_id[i]
                    color = colors[int(track_id) % len(colors)]
                    label = f'ID: {track_id}'
                else:
                    color = colors[i % len(colors)]
                    label = f'Det: {i}'
                
                # Desenha bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                detection_count += 1
    
    elif not SUPERVISION_AVAILABLE:
        for i, detection in enumerate(detections):
            x1, y1, x2, y2, conf = map(int, detection[:5])
            
            # Verifica se está dentro da ROI
            if (x1 >= roi_x and y1 >= roi_y and 
                x2 <= roi_x + roi_w and y2 <= roi_y + roi_h):
                
                color = colors[i % len(colors)]
                label = f'Det: {conf:.2f}'
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                detection_count += 1
    
    # Mostra contagem de detecções
    info_text = f'Deteccoes: {detection_count}'
    cv2.putText(frame, info_text, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame, detection_count

def zoomed_roi_tracking():
    """Função principal com ROI ampliada"""
    
    # Verificações iniciais
    if not os.path.exists(VIDEO_PATH):
        print(f"✗ Vídeo não encontrado: {VIDEO_PATH}")
        return
    
    if not os.path.exists(MODEL_PATH):
        print(f"✗ Modelo não encontrado: {MODEL_PATH}")
        return
    
    print("✓ Arquivos encontrados")
    
    # Carrega modelo
    print("Carregando modelo YOLO...")
    model = YOLO(MODEL_PATH)
    print("✓ Modelo YOLO carregado")
    
    # Inicializa tracker se disponível
    tracker = None
    if TRACKING_AVAILABLE:
        try:
            tracker = SORTTracker()
            print("✓ Tracker SORT inicializado")
        except Exception as e:
            print(f"⚠ Erro ao inicializar tracker: {e}")
    
    # Abre vídeo
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("✗ Erro ao abrir vídeo")
        return
    
    # Informações do vídeo
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"✓ Vídeo: {width}x{height}, {fps} FPS")
    print(f"ROI original: {ROI_WIDTH}x{ROI_HEIGHT}")
    
    # Calcula parâmetros de zoom
    zoom_info = None
    if TARGET_SIZE:
        zoom_info = TARGET_SIZE / max(ROI_WIDTH, ROI_HEIGHT)
        print(f"Zoom automático para {TARGET_SIZE}px: {zoom_info:.2f}x")
    elif ZOOM_FACTOR:
        zoom_info = ZOOM_FACTOR
        print(f"Zoom manual: {ZOOM_FACTOR}x")
    
    # Configura saída
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
    
    # Cores
    colors = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), 
        (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)
    ]
    
    # Processa frames
    max_frames = min(int(20 * fps), total_frames)
    frame_count = 0
    total_detections = 0
    
    print(f"Processando {max_frames} frames...")
    
    try:
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            progress = (frame_count + 1) / max_frames * 100
            print(f"Frame {frame_count + 1}/{max_frames} ({progress:.1f}%)", end='\r')
            
            # Extrai e amplia ROI
            roi_resized, roi_coords, scale_factor = extract_and_resize_roi(
                frame, ROI_X, ROI_Y, ROI_WIDTH, ROI_HEIGHT, 
                target_size=TARGET_SIZE, zoom_factor=ZOOM_FACTOR
            )
            
            # Executa detecção na ROI ampliada
            try:
                results = model(roi_resized, conf=0.3, verbose=False)
                
                if SUPERVISION_AVAILABLE:
                    detections = sv.Detections.from_ultralytics(results[0])
                    
                    if len(detections) > 0:
                        # Converte coordenadas para frame original
                        detections = convert_detections_to_original_frame(
                            detections, roi_coords, scale_factor
                        )
                        
                        # Aplica tracking se disponível
                        if tracker is not None:
                            try:
                                detections = tracker.update(detections)
                            except Exception as e:
                                print(f"\nErro no tracking: {e}")
                
                else:
                    # Versão sem supervision
                    detections = []
                    for result in results:
                        boxes = result.boxes
                        if boxes is not None:
                            for box in boxes:
                                x1, y1, x2, y2 = box.xyxy[0]
                                conf = float(box.conf[0])
                                detections.append([x1, y1, x2, y2, conf])
                    
                    if detections:
                        detections = convert_detections_to_original_frame(
                            detections, roi_coords, scale_factor
                        )
                
                # Desenha resultados
                annotated_frame = frame.copy()
                annotated_frame, det_count = draw_roi_and_detections(
                    annotated_frame, roi_coords, detections, colors, zoom_info
                )
                
                total_detections += det_count
                
                # Salva frame
                out.write(annotated_frame)
                
                # Mostra preview (opcional)
                cv2.imshow('Zoomed ROI Tracking', annotated_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            except Exception as e:
                print(f"\nErro no frame {frame_count}: {e}")
                # Salva frame com ROI apenas
                annotated_frame = frame.copy()
                cv2.rectangle(annotated_frame, (ROI_X, ROI_Y), 
                             (ROI_X + ROI_WIDTH, ROI_Y + ROI_HEIGHT), (255, 255, 255), 2)
                out.write(annotated_frame)
            
            frame_count += 1
        
        print(f"\n✓ Processamento concluído!")
        print(f"Frames processados: {frame_count}")
        print(f"Total de detecções: {total_detections}")
        print(f"Média de detecções por frame: {total_detections/frame_count:.2f}")
        print(f"Vídeo salvo em: {OUTPUT_PATH}")
        
    except KeyboardInterrupt:
        print("\n⚠ Interrompido pelo usuário")
    
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("=== Tracking com ROI Ampliada (Zoom) ===")
    print(f"ROI: {ROI_X}, {ROI_Y}, {ROI_WIDTH}x{ROI_HEIGHT}")
    print(f"Zoom: {ZOOM_FACTOR}x ou tamanho alvo: {TARGET_SIZE}px")
    print("Pressione 'q' para sair durante a reprodução")
    print()
    
    zoomed_roi_tracking()