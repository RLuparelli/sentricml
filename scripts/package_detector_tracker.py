import cv2
import numpy as np
from ultralytics import YOLO
import json
import os
from pathlib import Path
from datetime import datetime
import time
from collections import defaultdict, deque
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PackageTracker:
    """Rastreador de pacotes otimizado para esteira"""
    
    def __init__(self, max_disappeared=30, max_distance=50):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
    def register(self, centroid):
        """Registra um novo objeto"""
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1
        
    def deregister(self, object_id):
        """Remove um objeto do rastreamento"""
        del self.objects[object_id]
        del self.disappeared[object_id]
        
    def update(self, rects):
        """Atualiza o rastreamento com novas detecções"""
        if len(rects) == 0:
            # Marca todos os objetos como desaparecidos
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.get_objects()
        
        # Inicializa centroides das detecções
        input_centroids = np.zeros((len(rects), 2), dtype="int")
        
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            input_centroids[i] = (cX, cY)
        
        # Se não há objetos rastreados, registra todos
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i])
        else:
            # Calcula distâncias entre objetos existentes e novos centroides
            object_centroids = list(self.objects.values())
            object_ids = list(self.objects.keys())
            
            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)
            
            # Encontra os pares com menor distância
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_row_idxs = set()
            used_col_idxs = set()
            
            for (row, col) in zip(rows, cols):
                if row in used_row_idxs or col in used_col_idxs:
                    continue
                
                if D[row, col] > self.max_distance:
                    continue
                
                # Atualiza centroide do objeto
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                
                used_row_idxs.add(row)
                used_col_idxs.add(col)
            
            # Lida com objetos não pareados
            unused_row_idxs = set(range(0, D.shape[0])).difference(used_row_idxs)
            unused_col_idxs = set(range(0, D.shape[1])).difference(used_col_idxs)
            
            if D.shape[0] >= D.shape[1]:
                # Mais objetos que detecções
                for row in unused_row_idxs:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                # Mais detecções que objetos
                for col in unused_col_idxs:
                    self.register(input_centroids[col])
        
        return self.get_objects()
    
    def get_objects(self):
        """Retorna objetos ativos"""
        return self.objects

class PackageDetector:
    """Detector de pacotes em esteira com rastreamento"""
    
    def __init__(self, model_path, roi_path=None):
        self.model_path = Path(model_path)
        self.roi_path = Path(roi_path) if roi_path else None
        self.model = None
        self.roi_data = None
        self.tracker = PackageTracker()
        
        # Estatísticas
        self.stats = {
            'total_packages': 0,
            'packages_per_minute': deque(maxlen=60),
            'detection_history': deque(maxlen=1000)
        }
        
        # Configurações
        self.conf_threshold = 0.5
        self.nms_threshold = 0.4
        self.tracking_enabled = True
        
        self.load_model()
        self.load_roi()
    
    def load_model(self):
        """Carrega o modelo YOLO"""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Modelo não encontrado: {self.model_path}")
            
            self.model = YOLO(str(self.model_path))
            logger.info(f"✅ Modelo carregado: {self.model_path.name}")
            
            # Verifica classes do modelo
            if hasattr(self.model, 'names'):
                logger.info(f"📦 Classes detectáveis: {list(self.model.names.values())}")
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar modelo: {e}")
            raise
    
    def load_roi(self):
        """Carrega configuração de ROI"""
        if not self.roi_path or not self.roi_path.exists():
            logger.warning("⚠️ ROI não especificada, usando frame completo")
            return
        
        try:
            with open(self.roi_path, 'r', encoding='utf-8') as f:
                self.roi_data = json.load(f)
            
            logger.info(f"✅ ROI carregada: {self.roi_path.name}")
            logger.info(f"📍 Tipo: {self.roi_data['roi']['type']}")
            logger.info(f"📍 Pontos: {self.roi_data['roi']['points_count']}")
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar ROI: {e}")
            self.roi_data = None
    
    def is_point_in_roi(self, point):
        """Verifica se um ponto está dentro da ROI"""
        if not self.roi_data:
            return True
        
        roi_points = np.array(self.roi_data['roi']['points'])
        result = cv2.pointPolygonTest(roi_points, point, False)
        return result >= 0
    
    def is_detection_in_roi(self, bbox):
        """Verifica se uma detecção está dentro da ROI"""
        if not self.roi_data:
            return True
        
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        return self.is_point_in_roi((center_x, center_y))
    
    def detect_packages(self, frame):
        """Detecta pacotes no frame"""
        try:
            # Executa detecção
            results = self.model(frame, conf=self.conf_threshold, verbose=False)
            
            # Processa detecções
            detections = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # Coordenadas
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Verifica se está na ROI
                        if self.is_detection_in_roi([x1, y1, x2, y2]):
                            detections.append({
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': confidence,
                                'class_id': class_id,
                                'timestamp': time.time()
                            })
            
            return detections
            
        except Exception as e:
            logger.error(f"❌ Erro na detecção: {e}")
            return []
    
    def update_tracking(self, detections):
        """Atualiza rastreamento dos pacotes"""
        if not self.tracking_enabled:
            return detections
        
        # Converte detecções para formato do tracker
        rects = []
        for detection in detections:
            bbox = detection['bbox']
            rects.append((bbox[0], bbox[1], bbox[2], bbox[3]))
        
        # Atualiza tracker
        objects = self.tracker.update(rects)
        
        # Associa IDs às detecções
        tracked_detections = []
        for detection in detections:
            bbox = detection['bbox']
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            # Encontra o objeto mais próximo
            min_distance = float('inf')
            best_id = None
            
            for obj_id, centroid in objects.items():
                distance = np.sqrt((center_x - centroid[0])**2 + (center_y - centroid[1])**2)
                if distance < min_distance:
                    min_distance = distance
                    best_id = obj_id
            
            detection['track_id'] = best_id if min_distance < 50 else None
            tracked_detections.append(detection)
        
        return tracked_detections
    
    def update_stats(self, detections):
        """Atualiza estatísticas"""
        current_count = len(detections)
        current_time = time.time()
        
        # Atualiza contagem por minuto
        self.stats['packages_per_minute'].append({
            'time': current_time,
            'count': current_count
        })
        
        # Atualiza histórico
        self.stats['detection_history'].append({
            'timestamp': current_time,
            'count': current_count,
            'detections': detections
        })
        
        # Conta novos pacotes (tracking)
        new_packages = 0
        for detection in detections:
            if detection.get('track_id') is not None:
                # Lógica para contar apenas pacotes que cruzaram uma linha
                pass
        
        self.stats['total_packages'] += new_packages
    
    def draw_detections(self, frame, detections):
        """Desenha detecções no frame"""
        annotated_frame = frame.copy()
        
        # Desenha ROI se disponível
        if self.roi_data:
            roi_points = np.array(self.roi_data['roi']['points'])
            cv2.polylines(annotated_frame, [roi_points], True, (255, 255, 0), 2)
            cv2.putText(annotated_frame, f"ROI: {self.roi_data['roi']['type']}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Desenha detecções
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            track_id = detection.get('track_id')
            
            x1, y1, x2, y2 = bbox
            
            # Cor baseada no track_id
            if track_id is not None:
                color = self.get_color_for_id(track_id)
                label = f"ID:{track_id} {confidence:.2f}"
            else:
                color = (0, 255, 0)
                label = f"Det {confidence:.2f}"
            
            # Desenha bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Desenha label
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Desenha estatísticas
        self.draw_stats(annotated_frame, detections)
        
        return annotated_frame
    
    def get_color_for_id(self, track_id):
        """Gera cor consistente para um ID"""
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
            (255, 192, 203), (0, 128, 0), (128, 128, 0), (128, 0, 0)
        ]
        return colors[track_id % len(colors)]
    
    def draw_stats(self, frame, detections):
        """Desenha estatísticas na tela"""
        # Fundo para estatísticas
        overlay = frame.copy()
        cv2.rectangle(overlay, (frame.shape[1] - 300, 10), 
                     (frame.shape[1] - 10, 150), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Estatísticas
        stats_text = [
            f"Pacotes detectados: {len(detections)}",
            f"Total contados: {self.stats['total_packages']}",
            f"Confiança: {self.conf_threshold:.2f}",
            f"Rastreamento: {'ON' if self.tracking_enabled else 'OFF'}",
            f"Modelo: {self.model_path.name}"
        ]
        
        for i, text in enumerate(stats_text):
            y_pos = 30 + i * 20
            cv2.putText(frame, text, (frame.shape[1] - 295, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def process_video(self, video_path, output_path=None, max_frames=None):
        """Processa um vídeo detectando pacotes"""
        
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Vídeo não encontrado: {video_path}")
        
        # Abre vídeo
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Erro ao abrir vídeo: {video_path}")
        
        # Informações do vídeo
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"📹 Processando vídeo: {video_path.name}")
        logger.info(f"📊 Dimensões: {width}x{height}, {fps:.2f} FPS")
        logger.info(f"⏱️ Duração: {total_frames/fps:.2f}s ({total_frames} frames)")
        
        # Configura saída se especificada
        out = None
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            logger.info(f"💾 Salvando em: {output_path}")
        
        # Processa frames
        frame_count = 0
        max_frames = max_frames or total_frames
        start_time = time.time()
        
        try:
            while frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Progresso
                if frame_count % 30 == 0:  # A cada 30 frames
                    progress = (frame_count / max_frames) * 100
                    elapsed = time.time() - start_time
                    fps_processing = frame_count / elapsed if elapsed > 0 else 0
                    
                    logger.info(f"🎬 Frame {frame_count}/{max_frames} "
                              f"({progress:.1f}%) - {fps_processing:.1f} FPS")
                
                # Detecta pacotes
                detections = self.detect_packages(frame)
                
                # Atualiza rastreamento
                if self.tracking_enabled:
                    detections = self.update_tracking(detections)
                
                # Atualiza estatísticas
                self.update_stats(detections)
                
                # Desenha resultados
                annotated_frame = self.draw_detections(frame, detections)
                
                # Salva frame se necessário
                if out:
                    out.write(annotated_frame)
                
                # Mostra preview (opcional)
                if frame_count % 10 == 0:  # A cada 10 frames
                    cv2.imshow('Package Detection', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("⚠️ Processamento interrompido pelo usuário")
                        break
                
                frame_count += 1
            
            # Estatísticas finais
            elapsed = time.time() - start_time
            avg_fps = frame_count / elapsed if elapsed > 0 else 0
            
            logger.info(f"✅ Processamento concluído!")
            logger.info(f"📊 Frames processados: {frame_count}")
            logger.info(f"⏱️ Tempo total: {elapsed:.2f}s")
            logger.info(f"🎬 FPS médio: {avg_fps:.2f}")
            logger.info(f"📦 Total de pacotes detectados: {self.stats['total_packages']}")
            
        except KeyboardInterrupt:
            logger.info("⚠️ Processamento interrompido")
        
        finally:
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
        
        return {
            'frames_processed': frame_count,
            'total_packages': self.stats['total_packages'],
            'processing_time': elapsed,
            'average_fps': avg_fps
        }

def main():
    """Função principal"""
    
    print("🚚 DETECTOR DE PACOTES EM ESTEIRA")
    print("="*50)
    
    # Caminhos
    model_path = Path(r"D:\Sentric\MercadoLivre\dataset3.0\sentricml\models\MercadoLivreBest.pt")
    videos_dir = Path(r"D:\Sentric\MercadoLivre\dataset3.0\sentricml\videos")
    roi_dir = Path(r"D:\Sentric\MercadoLivre\dataset3.0\sentricml\roi")
    output_dir = Path(r"D:\Sentric\MercadoLivre\dataset3.0\sentricml\output")
    
    # Verifica modelo
    if not model_path.exists():
        print(f"❌ Modelo não encontrado: {model_path}")
        return
    
    # Lista vídeos disponíveis
    videos = []
    if videos_dir.exists():
        videos = list(videos_dir.glob("*.mp4")) + list(videos_dir.glob("*.avi"))
    
    if not videos:
        print("❌ Nenhum vídeo encontrado")
        video_path = input("Digite o caminho do vídeo: ").strip()
        if not video_path:
            return
        video_path = Path(video_path)
    else:
        print(f"🎬 Vídeos encontrados ({len(videos)}):")
        for i, video in enumerate(videos, 1):
            print(f"   {i}. {video.name}")
        
        choice = input("Escolha um vídeo (número) ou digite caminho: ").strip()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(videos):
                video_path = videos[idx]
            else:
                print("❌ Escolha inválida")
                return
        else:
            video_path = Path(choice) if choice else videos[0]
    
    # Lista ROIs disponíveis
    rois = []
    if roi_dir.exists():
        rois = list(roi_dir.glob("*.json"))
    
    roi_path = None
    if rois:
        print(f"\n📍 ROIs encontradas ({len(rois)}):")
        print("   0. Sem ROI (frame completo)")
        for i, roi in enumerate(rois, 1):
            print(f"   {i}. {roi.name}")
        
        choice = input("Escolha uma ROI (número): ").strip()
        if choice.isdigit():
            idx = int(choice)
            if idx == 0:
                roi_path = None
            elif 1 <= idx <= len(rois):
                roi_path = rois[idx - 1]
    
    # Configurações
    print(f"\n⚙️ Configurações:")
    print(f"   🎬 Vídeo: {video_path.name}")
    print(f"   📍 ROI: {roi_path.name if roi_path else 'Frame completo'}")
    print(f"   🤖 Modelo: {model_path.name}")
    
    # Cria detector
    try:
        detector = PackageDetector(model_path, roi_path)
        
        # Configura saída
        output_path = output_dir / f"{video_path.stem}_detected.mp4"
        
        # Processa vídeo
        max_frames = input("Máximo de frames (Enter para todos): ").strip()
        max_frames = int(max_frames) if max_frames.isdigit() else None
        
        results = detector.process_video(video_path, output_path, max_frames)
        
        print(f"\n🎉 Processamento concluído!")
        print(f"📊 Resultados:")
        print(f"   - Frames processados: {results['frames_processed']}")
        print(f"   - Pacotes detectados: {results['total_packages']}")
        print(f"   - Tempo de processamento: {results['processing_time']:.2f}s")
        print(f"   - FPS médio: {results['average_fps']:.2f}")
        
    except Exception as e:
        logger.error(f"❌ Erro: {e}")

if __name__ == "__main__":
    main()