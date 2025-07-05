import cv2
import numpy as np
from ultralytics import YOLO
import os

# Primeiro vamos testar se conseguimos importar as bibliotecas
try:
    import supervision as sv
    print("✓ Supervision importado com sucesso")
except ImportError:
    print("✗ Erro ao importar supervision. Instale com: pip install supervision")
    exit(1)

try:
    from trackers import SORTTracker
    print("✓ SORTTracker importado com sucesso")
except ImportError:
    print("✗ Erro ao importar trackers. Instale com: pip install trackers")
    exit(1)

# Configuração dos caminhos
VIDEO_PATH = r"C:\Users\lupar\Desktop\Sentric\MercadoLivre\Scripts\videos\esteiraML.mp4"
MODEL_PATH = r"C:\Users\lupar\Desktop\Sentric\MercadoLivre\Scripts\MercadoLivreBest.pt"
OUTPUT_PATH = r"C:\Users\lupar\Desktop\Sentric\MercadoLivre\Scripts\videos\esteiraML_tracked.mp4"

# Configurações
CONFIDENCE_THRESHOLD = 0.3
NMS_THRESHOLD = 0.3
SAMPLE_DURATION = 360  # segundos

def test_tracker_api():
    """Testa a API do SORTTracker para ver quais parâmetros aceita"""
    try:
        # Teste básico
        tracker = SORTTracker()
        print("✓ SORTTracker criado com parâmetros padrão")
        return tracker
    except Exception as e:
        print(f"✗ Erro ao criar SORTTracker: {e}")
        return None

def setup_visualization():
    """Configura os elementos de visualização"""
    try:
        # Cores para diferentes tracks
        colors = [
            (255, 255, 0),   # Amarelo
            (255, 155, 0),   # Laranja
            (255, 128, 128), # Rosa claro
            (255, 102, 178), # Rosa
            (255, 102, 255), # Magenta
            (178, 102, 255), # Roxo claro
            (153, 153, 255), # Azul claro
            (51, 153, 255),  # Azul
            (102, 255, 255), # Ciano
            (51, 255, 153),  # Verde claro
            (102, 255, 102), # Verde
            (153, 255, 0),   # Verde lima
        ]
        
        return colors
    except Exception as e:
        print(f"Erro na configuração de visualização: {e}")
        return [(0, 255, 0)] * 12  # Verde padrão

def draw_tracking_results(frame, detections, colors):
    """Desenha os resultados do tracking no frame"""
    annotated_frame = frame.copy()
    
    if len(detections) == 0:
        return annotated_frame
    
    # Se não tem tracker_id, desenha apenas as detecções
    if not hasattr(detections, 'tracker_id') or detections.tracker_id is None:
        for bbox in detections.xyxy:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return annotated_frame
    
    # Desenha com tracker IDs
    for i, (bbox, track_id) in enumerate(zip(detections.xyxy, detections.tracker_id)):
        x1, y1, x2, y2 = map(int, bbox)
        
        # Escolhe cor baseada no ID
        color = colors[int(track_id) % len(colors)]
        
        # Desenha bounding box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        
        # Desenha ID
        label = f"ID: {int(track_id)}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        
        # Fundo para o texto
        cv2.rectangle(annotated_frame, 
                     (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), 
                     color, -1)
        
        # Texto
        cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    return annotated_frame

def main():
    # Verificações iniciais
    if not os.path.exists(VIDEO_PATH):
        print(f"✗ Vídeo não encontrado: {VIDEO_PATH}")
        return
    
    if not os.path.exists(MODEL_PATH):
        print(f"✗ Modelo não encontrado: {MODEL_PATH}")
        return
    
    print("✓ Arquivos encontrados")
    
    # Testa o tracker
    tracker = test_tracker_api()
    if tracker is None:
        return
    
    # Carrega modelo
    print("Carregando modelo YOLO...")
    try:
        model = YOLO(MODEL_PATH)
        print("✓ Modelo YOLO carregado")
    except Exception as e:
        print(f"✗ Erro ao carregar modelo: {e}")
        return
    
    # Configura visualização
    colors = setup_visualization()
    
    # Abre vídeo
    print("Abrindo vídeo...")
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print("✗ Erro ao abrir vídeo")
        return
    
    # Informações do vídeo
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"✓ Vídeo: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Calcula frames para processar
    max_frames = min(int(SAMPLE_DURATION * fps), total_frames)
    print(f"Processando {max_frames} frames ({SAMPLE_DURATION}s)")
    
    # Configura writer de saída
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
    
    frame_count = 0
    
    try:
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Progresso
            progress = (frame_count + 1) / max_frames * 100
            print(f"Processando: {progress:.1f}% ({frame_count + 1}/{max_frames})", end='\r')
            
            # Detecção YOLO
            try:
                results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
                detections = sv.Detections.from_ultralytics(results[0])
                
                # Aplica NMS
                if len(detections) > 0:
                    detections = detections.with_nms(threshold=NMS_THRESHOLD)
                
                # Update tracker
                if len(detections) > 0:
                    try:
                        detections = tracker.update(detections)
                    except Exception as e:
                        print(f"\nErro no tracker: {e}")
                        # Continua sem tracking se houver erro
                        pass
                
                # Desenha resultados
                annotated_frame = draw_tracking_results(frame, detections, colors)
                
                # Salva frame
                out.write(annotated_frame)
                
            except Exception as e:
                print(f"\nErro no frame {frame_count}: {e}")
                # Salva frame original se houver erro
                out.write(frame)
            
            frame_count += 1
        
        print(f"\n✓ Processamento concluído!")
        print(f"Vídeo salvo em: {OUTPUT_PATH}")
        
    except KeyboardInterrupt:
        print("\n⚠ Processamento interrompido pelo usuário")
    
    except Exception as e:
        print(f"\n✗ Erro durante processamento: {e}")
    
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("=== SORT Tracking com YOLO Personalizado ===")
    print()
    
    # Verifica ambiente virtual
    import sys
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✓ Ambiente virtual ativo")
    else:
        print("⚠ Não está em ambiente virtual (recomendado)")
    
    print()
    
    try:
        main()
    except Exception as e:
        print(f"✗ Erro fatal: {e}")
        input("Pressione Enter para sair...")