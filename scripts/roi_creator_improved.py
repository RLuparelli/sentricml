import cv2
import numpy as np
from PIL import Image
import json
import os
from pathlib import Path
from datetime import datetime

class ROICreator:
    """Criador de ROI otimizado para detecção de pacotes em esteira"""
    
    def __init__(self):
        self.drawing = False
        self.roi_points = []
        self.temp_points = []
        self.image = None
        self.original_image = None
        self.window_name = "ROI Creator - Detecção de Pacotes"
        self.scale_factor = 1.0
        self.roi_types = {
            'counting_line': {'color': (0, 255, 0), 'thickness': 3},
            'detection_area': {'color': (255, 0, 0), 'thickness': 2},
            'exclusion_zone': {'color': (0, 0, 255), 'thickness': 2}
        }
        self.current_roi_type = 'detection_area'
        
    def mouse_callback(self, event, x, y, flags, param):
        """Callback para capturar eventos do mouse"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if not self.drawing:
                self.drawing = True
                self.temp_points = [(x, y)]
                print(f"🎯 Iniciando ROI - Ponto 1: ({x}, {y})")
            else:
                self.temp_points.append((x, y))
                print(f"📍 Ponto {len(self.temp_points)}: ({x}, {y})")
            
            self.update_display()
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Clique direito finaliza a ROI
            if len(self.temp_points) >= 3:
                self.roi_points = self.temp_points.copy()
                self.drawing = False
                self.draw_final_roi()
                print(f"✅ ROI finalizada com {len(self.roi_points)} pontos!")
                print(f"📦 Tipo: {self.current_roi_type}")
            else:
                print("❌ Defina pelo menos 3 pontos para criar uma ROI!")
            
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            # Mostra preview da linha
            self.update_display()
            if len(self.temp_points) > 0:
                color = self.roi_types[self.current_roi_type]['color']
                cv2.line(self.image, self.temp_points[-1], (x, y), color, 1)
                cv2.imshow(self.window_name, self.image)
    
    def update_display(self):
        """Atualiza a exibição da imagem"""
        self.image = self.original_image.copy()
        
        # Desenha pontos e linhas temporárias
        if len(self.temp_points) > 0:
            color = self.roi_types[self.current_roi_type]['color']
            thickness = self.roi_types[self.current_roi_type]['thickness']
            
            # Desenha pontos
            for i, point in enumerate(self.temp_points):
                cv2.circle(self.image, point, 6, color, -1)
                cv2.putText(self.image, str(i+1), 
                           (point[0]+10, point[1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Desenha linhas
            if len(self.temp_points) > 1:
                for i in range(len(self.temp_points) - 1):
                    cv2.line(self.image, self.temp_points[i], 
                            self.temp_points[i + 1], color, thickness)
        
        # Adiciona informações na tela
        self.add_info_overlay()
        cv2.imshow(self.window_name, self.image)
    
    def add_info_overlay(self):
        """Adiciona informações na tela"""
        # Fundo para as informações
        info_bg = np.zeros((150, 400, 3), dtype=np.uint8)
        
        # Informações
        info_lines = [
            f"Tipo de ROI: {self.current_roi_type}",
            f"Pontos: {len(self.temp_points)}",
            "",
            "Controles:",
            "• Clique esquerdo: Adicionar ponto",
            "• Clique direito: Finalizar ROI",
            "• T: Mudar tipo de ROI",
            "• R: Resetar",
            "• Enter: Salvar",
            "• Esc: Cancelar"
        ]
        
        for i, line in enumerate(info_lines):
            y_pos = 20 + i * 15
            color = (255, 255, 255) if line else (100, 100, 100)
            cv2.putText(info_bg, line, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Sobrepor na imagem
        self.image[10:160, 10:410] = info_bg
    
    def draw_final_roi(self):
        """Desenha a ROI final"""
        if len(self.roi_points) >= 3:
            color = self.roi_types[self.current_roi_type]['color']
            thickness = self.roi_types[self.current_roi_type]['thickness']
            
            # Cria overlay semi-transparente
            overlay = self.original_image.copy()
            cv2.fillPoly(overlay, [np.array(self.roi_points)], color)
            self.image = cv2.addWeighted(self.original_image, 0.7, overlay, 0.3, 0)
            
            # Desenha contorno
            cv2.polylines(self.image, [np.array(self.roi_points)], True, color, thickness)
            
            # Desenha pontos numerados
            for i, point in enumerate(self.roi_points):
                cv2.circle(self.image, point, 8, (255, 255, 255), -1)
                cv2.circle(self.image, point, 6, color, -1)
                cv2.putText(self.image, str(i+1), 
                           (point[0]+15, point[1]-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Adiciona informação da ROI
            roi_info = f"ROI {self.current_roi_type.upper()} - {len(self.roi_points)} pontos"
            cv2.putText(self.image, roi_info, (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(self.image, "Pressione ENTER para salvar", (10, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow(self.window_name, self.image)
    
    def create_roi_from_image(self, image_path, roi_save_path=None):
        """Cria ROI a partir de uma imagem"""
        
        # Carrega imagem
        image_path = Path(image_path)
        if not image_path.exists():
            print(f"❌ Imagem não encontrada: {image_path}")
            return None
            
        self.original_image = cv2.imread(str(image_path))
        if self.original_image is None:
            # Tenta com PIL para formatos especiais
            try:
                pil_image = Image.open(image_path)
                self.original_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"❌ Erro ao carregar imagem: {e}")
                return None
        
        # Redimensiona se muito grande
        height, width = self.original_image.shape[:2]
        self.scale_factor = 1.0
        max_dim = 1200
        
        if max(height, width) > max_dim:
            self.scale_factor = max_dim / max(height, width)
            new_width = int(width * self.scale_factor)
            new_height = int(height * self.scale_factor)
            self.original_image = cv2.resize(self.original_image, (new_width, new_height))
            print(f"📏 Imagem redimensionada para: {new_width}x{new_height}")
        
        self.image = self.original_image.copy()
        
        # Configura janela
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        # Mostra instruções
        self.show_instructions()
        self.update_display()
        
        # Loop principal
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == 13:  # Enter - Salvar
                if len(self.roi_points) >= 3:
                    # Ajusta coordenadas se imagem foi redimensionada
                    final_roi = self.roi_points.copy()
                    if self.scale_factor != 1.0:
                        final_roi = [(int(x/self.scale_factor), int(y/self.scale_factor)) 
                                   for x, y in final_roi]
                    
                    # Salva ROI
                    if roi_save_path:
                        saved_path = self.save_roi(final_roi, roi_save_path, image_path)
                        if saved_path:
                            cv2.destroyAllWindows()
                            return saved_path
                    else:
                        cv2.destroyAllWindows()
                        return final_roi
                else:
                    print("❌ Defina pelo menos 3 pontos para criar uma ROI!")
            
            elif key == 27:  # Esc - Cancelar
                print("❌ Criação de ROI cancelada")
                cv2.destroyAllWindows()
                return None
            
            elif key == ord('r') or key == ord('R'):  # R - Reset
                self.reset_roi()
            
            elif key == ord('t') or key == ord('T'):  # T - Mudar tipo
                self.cycle_roi_type()
            
            elif key == ord('h') or key == ord('H'):  # H - Help
                self.show_instructions()
        
        cv2.destroyAllWindows()
        return None
    
    def cycle_roi_type(self):
        """Alterna entre tipos de ROI"""
        types = list(self.roi_types.keys())
        current_index = types.index(self.current_roi_type)
        next_index = (current_index + 1) % len(types)
        self.current_roi_type = types[next_index]
        
        print(f"🔄 Tipo de ROI alterado para: {self.current_roi_type}")
        self.update_display()
    
    def reset_roi(self):
        """Reseta a ROI"""
        self.roi_points = []
        self.temp_points = []
        self.drawing = False
        self.update_display()
        print("🔄 ROI resetada!")
    
    def show_instructions(self):
        """Mostra instruções de uso"""
        print("\n" + "="*80)
        print("🎯 ROI CREATOR - DETECÇÃO DE PACOTES EM ESTEIRA")
        print("="*80)
        print("📦 TIPOS DE ROI:")
        print("   • detection_area: Área principal de detecção")
        print("   • counting_line: Linha de contagem de pacotes")
        print("   • exclusion_zone: Zona a ser ignorada")
        print()
        print("🎮 CONTROLES:")
        print("   • Clique ESQUERDO: Adicionar ponto")
        print("   • Clique DIREITO: Finalizar ROI")
        print("   • T: Mudar tipo de ROI")
        print("   • R: Resetar ROI")
        print("   • Enter: Salvar ROI")
        print("   • Esc: Cancelar")
        print("   • H: Mostrar ajuda")
        print("="*80)
    
    def save_roi(self, roi_points, save_path, image_path):
        """Salva ROI em formato JSON otimizado"""
        try:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Dados da ROI
            roi_data = {
                'metadata': {
                    'created_at': datetime.now().isoformat(),
                    'source_image': str(image_path),
                    'image_dimensions': {
                        'width': self.original_image.shape[1],
                        'height': self.original_image.shape[0]
                    },
                    'roi_type': self.current_roi_type,
                    'scale_factor': self.scale_factor
                },
                'roi': {
                    'points': roi_points,
                    'points_count': len(roi_points),
                    'type': self.current_roi_type
                },
                'config': {
                    'for_package_detection': True,
                    'conveyor_belt_setup': True,
                    'yolo_compatible': True
                }
            }
            
            # Salva JSON
            json_path = save_path.with_suffix('.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(roi_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ ROI salva em: {json_path}")
            print(f"📦 Tipo: {self.current_roi_type}")
            print(f"📍 Pontos: {len(roi_points)}")
            
            return json_path
            
        except Exception as e:
            print(f"❌ Erro ao salvar ROI: {e}")
            return None

def load_roi_from_json(json_path):
    """Carrega ROI de um arquivo JSON"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            roi_data = json.load(f)
        
        print(f"✅ ROI carregada de: {json_path}")
        print(f"📦 Tipo: {roi_data['roi']['type']}")
        print(f"📍 Pontos: {roi_data['roi']['points_count']}")
        
        return roi_data
        
    except Exception as e:
        print(f"❌ Erro ao carregar ROI: {e}")
        return None

def main():
    """Função principal"""
    
    print("🎯 ROI CREATOR - DETECÇÃO DE PACOTES")
    print("="*50)
    
    # Caminhos padrão
    photos_dir = Path(r"D:\Sentric\MercadoLivre\dataset3.0\sentricml\photos")
    roi_dir = Path(r"D:\Sentric\MercadoLivre\dataset3.0\sentricml\roi")
    
    # Verifica se existem fotos
    if photos_dir.exists():
        photos = list(photos_dir.glob("*.jpg")) + list(photos_dir.glob("*.png"))
        
        if photos:
            print(f"📸 Fotos encontradas ({len(photos)}):")
            for i, photo in enumerate(photos[:10], 1):  # Mostra até 10
                print(f"   {i}. {photo.name}")
            if len(photos) > 10:
                print(f"   ... e mais {len(photos) - 10} fotos")
            print()
            
            # Usar a foto mais recente ou permitir escolha
            latest_photo = max(photos, key=lambda p: p.stat().st_mtime)
            print(f"🎯 Foto mais recente: {latest_photo.name}")
            
            # Escolher foto
            choice = input("Usar foto mais recente? (s/N) ou digite o número da foto: ").strip()
            
            if choice.lower() in ['s', 'sim', 'y', 'yes']:
                selected_photo = latest_photo
            elif choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(photos):
                    selected_photo = photos[idx]
                else:
                    print("❌ Número inválido")
                    return
            else:
                # Permitir caminho manual
                photo_path = input("Digite o caminho da foto: ").strip()
                if photo_path:
                    selected_photo = Path(photo_path)
                else:
                    selected_photo = latest_photo
        else:
            print("❌ Nenhuma foto encontrada na pasta padrão")
            photo_path = input("Digite o caminho da foto: ").strip()
            if not photo_path:
                print("❌ Caminho da foto é obrigatório")
                return
            selected_photo = Path(photo_path)
    else:
        print("❌ Pasta de fotos não encontrada")
        photo_path = input("Digite o caminho da foto: ").strip()
        if not photo_path:
            print("❌ Caminho da foto é obrigatório")
            return
        selected_photo = Path(photo_path)
    
    # Verifica se a foto existe
    if not selected_photo.exists():
        print(f"❌ Foto não encontrada: {selected_photo}")
        return
    
    print(f"📸 Usando foto: {selected_photo.name}")
    
    # Define caminho de saída da ROI
    roi_name = selected_photo.stem + "_roi"
    roi_save_path = roi_dir / roi_name
    
    # Cria o criador de ROI
    roi_creator = ROICreator()
    
    # Cria a ROI
    roi_path = roi_creator.create_roi_from_image(selected_photo, roi_save_path)
    
    if roi_path:
        print(f"\n✅ ROI criada com sucesso!")
        print(f"📁 Salva em: {roi_path}")
        print(f"🎯 Pronta para usar na detecção de pacotes")
    else:
        print("\n❌ Nenhuma ROI foi criada")

if __name__ == "__main__":
    main()