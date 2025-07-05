import cv2
import numpy as np
from PIL import Image
import pickle
import os
from pathlib import Path

class SimpleROICreator:
    """Ferramenta simples para criar e salvar ROI"""
    
    def __init__(self):
        self.drawing = False
        self.roi_points = []
        self.temp_points = []
        self.image = None
        self.original_image = None
        self.window_name = "Criar ROI - Região de Interesse"
        
    def mouse_callback(self, event, x, y, flags, param):
        """Callback para capturar cliques do mouse"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if not self.drawing:
                self.drawing = True
                self.temp_points = [(x, y)]
                print(f"Ponto 1: ({x}, {y})")
            else:
                self.temp_points.append((x, y))
                print(f"Ponto {len(self.temp_points)}: ({x}, {y})")
            
            # Desenha ponto
            cv2.circle(self.image, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow(self.window_name, self.image)
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Clique direito finaliza o polígono
            if len(self.temp_points) >= 3:
                self.roi_points = self.temp_points.copy()
                self.drawing = False
                self.draw_final_roi()
                print(f"✓ ROI finalizada com {len(self.roi_points)} pontos!")
            else:
                print("❌ Defina pelo menos 3 pontos para criar uma ROI!")
            
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            # Mostra preview da linha
            temp_image = self.original_image.copy()
            
            # Desenha pontos já marcados
            for i, point in enumerate(self.temp_points):
                cv2.circle(temp_image, point, 5, (0, 255, 0), -1)
                cv2.putText(temp_image, str(i+1), 
                           (point[0]+10, point[1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Desenha linhas entre pontos
            if len(self.temp_points) > 1:
                for i in range(len(self.temp_points) - 1):
                    cv2.line(temp_image, self.temp_points[i], self.temp_points[i + 1], (0, 255, 0), 2)
            
            # Linha temporária até o cursor
            if len(self.temp_points) > 0:
                cv2.line(temp_image, self.temp_points[-1], (x, y), (0, 255, 255), 1)
            
            self.image = temp_image
            cv2.imshow(self.window_name, self.image)
    
    def draw_final_roi(self):
        """Desenha a ROI final"""
        if len(self.roi_points) >= 3:
            # Cria overlay semi-transparente
            overlay = self.original_image.copy()
            cv2.fillPoly(overlay, [np.array(self.roi_points)], (0, 255, 0))
            self.image = cv2.addWeighted(self.original_image, 0.7, overlay, 0.3, 0)
            
            # Desenha contorno
            cv2.polylines(self.image, [np.array(self.roi_points)], True, (0, 255, 0), 3)
            
            # Desenha pontos numerados
            for i, point in enumerate(self.roi_points):
                cv2.circle(self.image, point, 8, (0, 0, 255), -1)
                cv2.putText(self.image, str(i+1), 
                           (point[0]+15, point[1]-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Adiciona texto informativo
            cv2.putText(self.image, "ROI DEFINIDA - Pressione ENTER para salvar", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow(self.window_name, self.image)
    
    def create_roi(self, image_path, roi_save_path=None):
        """Interface principal para criar ROI"""
        
        # Carrega imagem
        if isinstance(image_path, str):
            image_path = Path(image_path)
            
        self.original_image = cv2.imread(str(image_path))
        if self.original_image is None:
            # Tenta com PIL para WEBP
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
        
        cv2.imshow(self.window_name, self.image)
        
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
                        self.save_roi(final_roi, roi_save_path)
                    
                    cv2.destroyAllWindows()
                    return final_roi
                else:
                    print("❌ Defina pelo menos 3 pontos para criar uma ROI!")
            
            elif key == 27:  # Esc - Cancelar
                print("❌ Criação de ROI cancelada")
                cv2.destroyAllWindows()
                return None
            
            elif key == ord('r'):  # R - Reset
                self.reset_roi()
            
            elif key == ord('h'):  # H - Help
                self.show_instructions()
        
        cv2.destroyAllWindows()
        return None
    
    def reset_roi(self):
        """Reseta a ROI"""
        self.roi_points = []
        self.temp_points = []
        self.drawing = False
        self.image = self.original_image.copy()
        cv2.imshow(self.window_name, self.image)
        print("🔄 ROI resetada!")
    
    def show_instructions(self):
        """Mostra instruções de uso"""
        print("\n" + "="*70)
        print("🎯 CRIADOR DE ROI - MÚLTIPLOS POLÍGONOS")
        print("="*70)
        print("CONTROLES:")
        print("• Clique ESQUERDO: Adicionar ponto ao polígono atual")
        print("• Clique DIREITO: Finalizar polígono atual")
        print("• 'N': Iniciar novo polígono")
        print("• 'R': Resetar tudo")
        print("• 'U': Desfazer último polígono")
        print("• 'Enter': Salvar e sair")
        print("• 'Esc': Cancelar")
        print("• 'H': Mostrar ajuda")
        print("="*70)
    
    def save_roi(self, roi_polygons, save_path):
        """Salva múltiplos polígonos ROI"""
        try:
            save_path = Path(save_path)
            
            # Salva em formato pickle
            with open(save_path.with_suffix('.pkl'), 'wb') as f:
                pickle.dump(roi_polygons, f)
            
            # Salva também em JSON para visualização
            import json
            json_data = {
                'total_polygons': len(roi_polygons),
                'polygons': [{'id': i+1, 'points': poly} for i, poly in enumerate(roi_polygons)]
            }
            
            with open(save_path.with_suffix('.json'), 'w') as f:
                json.dump(json_data, f, indent=2)
            
            print(f"✅ ROI salva em: {save_path.with_suffix('.pkl')}")
            print(f"✅ JSON salvo em: {save_path.with_suffix('.json')}")
            
        except Exception as e:
            print(f"❌ Erro ao salvar ROI: {e}")


class MultiROICreator:
    """Criador de múltiplas ROIs (polígonos) em uma imagem"""
    
    def __init__(self):
        self.drawing = False
        self.current_polygon = []
        self.completed_polygons = []
        self.temp_points = []
        self.image = None
        self.original_image = None
        self.window_name = "Criar Múltiplas ROIs"
        self.scale_factor = 1.0
        self.colors = [
            (0, 255, 0),    # Verde
            (255, 0, 0),    # Azul
            (0, 0, 255),    # Vermelho
            (255, 255, 0),  # Ciano
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Amarelo
            (128, 0, 128),  # Roxo
            (255, 165, 0),  # Laranja
        ]
        
    def mouse_callback(self, event, x, y, flags, param):
        """Callback para capturar cliques do mouse"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Adiciona ponto ao polígono atual
            self.current_polygon.append((x, y))
            print(f"Polígono {len(self.completed_polygons)+1}, Ponto {len(self.current_polygon)}: ({x}, {y})")
            self.redraw_image()
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Finaliza polígono atual
            if len(self.current_polygon) >= 3:
                self.completed_polygons.append(self.current_polygon.copy())
                print(f"✅ Polígono {len(self.completed_polygons)} finalizado com {len(self.current_polygon)} pontos!")
                self.current_polygon = []
                self.drawing = False
                self.redraw_image()
            else:
                print("❌ Defina pelo menos 3 pontos para finalizar o polígono!")
                
        elif event == cv2.EVENT_MOUSEMOVE:
            # Mostra preview da linha
            if len(self.current_polygon) > 0:
                self.redraw_image()
                # Linha temporária até o cursor
                color = self.get_color(len(self.completed_polygons))
                cv2.line(self.image, self.current_polygon[-1], (x, y), color, 1)
                cv2.imshow(self.window_name, self.image)
    
    def get_color(self, polygon_index):
        """Retorna cor para o polígono baseado no índice"""
        return self.colors[polygon_index % len(self.colors)]
    
    def redraw_image(self):
        """Redesenha a imagem com todos os polígonos"""
        self.image = self.original_image.copy()
        
        # Desenha polígonos completos
        for i, polygon in enumerate(self.completed_polygons):
            color = self.get_color(i)
            
            # Preenche polígono com transparência
            overlay = self.image.copy()
            cv2.fillPoly(overlay, [np.array(polygon)], color)
            self.image = cv2.addWeighted(self.image, 0.8, overlay, 0.2, 0)
            
            # Contorno
            cv2.polylines(self.image, [np.array(polygon)], True, color, 3)
            
            # Pontos numerados
            for j, point in enumerate(polygon):
                cv2.circle(self.image, point, 6, color, -1)
                cv2.putText(self.image, f"{i+1}.{j+1}", 
                           (point[0]+10, point[1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Label do polígono
            if polygon:
                center = np.mean(polygon, axis=0).astype(int)
                cv2.putText(self.image, f"ROI {i+1}", 
                           (center[0]-20, center[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Desenha polígono atual em construção
        if len(self.current_polygon) > 0:
            color = self.get_color(len(self.completed_polygons))
            
            # Pontos
            for j, point in enumerate(self.current_polygon):
                cv2.circle(self.image, point, 6, color, -1)
                cv2.putText(self.image, str(j+1), 
                           (point[0]+10, point[1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Linhas
            if len(self.current_polygon) > 1:
                for j in range(len(self.current_polygon) - 1):
                    cv2.line(self.image, self.current_polygon[j], self.current_polygon[j + 1], color, 2)
        
        # Info no topo
        info_text = f"Polígonos: {len(self.completed_polygons)} | Atual: {len(self.current_polygon)} pontos"
        cv2.putText(self.image, info_text, (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow(self.window_name, self.image)
    
    def create_multiple_rois(self, image_path, roi_save_path=None):
        """Interface principal para criar múltiplas ROIs"""
        
        # Carrega imagem
        if isinstance(image_path, str):
            image_path = Path(image_path)
            
        self.original_image = cv2.imread(str(image_path))
        if self.original_image is None:
            try:
                pil_image = Image.open(image_path)
                self.original_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"❌ Erro ao carregar imagem: {e}")
                return None
        
        # Redimensiona se necessário
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
        self.redraw_image()
        
        # Loop principal
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == 13:  # Enter - Salvar
                if len(self.completed_polygons) > 0:
                    # Ajusta coordenadas se imagem foi redimensionada
                    final_rois = []
                    for polygon in self.completed_polygons:
                        if self.scale_factor != 1.0:
                            adjusted_polygon = [(int(x/self.scale_factor), int(y/self.scale_factor)) 
                                              for x, y in polygon]
                        else:
                            adjusted_polygon = polygon
                        final_rois.append(adjusted_polygon)
                    
                    # Salva ROIs
                    if roi_save_path:
                        self.save_multiple_rois(final_rois, roi_save_path)
                    
                    cv2.destroyAllWindows()
                    return final_rois
                else:
                    print("❌ Crie pelo menos um polígono!")
            
            elif key == 27:  # Esc - Cancelar
                print("❌ Criação de ROIs cancelada")
                cv2.destroyAllWindows()
                return None
            
            elif key == ord('r'):  # R - Reset tudo
                self.reset_all()
            
            elif key == ord('u'):  # U - Desfazer último polígono
                self.undo_last_polygon()
            
            elif key == ord('n'):  # N - Novo polígono
                self.start_new_polygon()
            
            elif key == ord('h'):  # H - Help
                self.show_instructions()
        
        cv2.destroyAllWindows()
        return None
    
    def reset_all(self):
        """Reseta tudo"""
        self.completed_polygons = []
        self.current_polygon = []
        self.drawing = False
        self.redraw_image()
        print("🔄 Todos os polígonos resetados!")
    
    def undo_last_polygon(self):
        """Desfaz o último polígono"""
        if self.completed_polygons:
            removed = self.completed_polygons.pop()
            self.redraw_image()
            print(f"↩️ Último polígono removido ({len(removed)} pontos)")
        else:
            print("❌ Nenhum polígono para remover!")
    
    def start_new_polygon(self):
        """Inicia um novo polígono"""
        if len(self.current_polygon) >= 3:
            # Finaliza o atual primeiro
            self.completed_polygons.append(self.current_polygon.copy())
            print(f"✅ Polígono {len(self.completed_polygons)} auto-finalizado")
        
        self.current_polygon = []
        self.drawing = False
        self.redraw_image()
        print(f"🆕 Iniciando polígono {len(self.completed_polygons)+1}")
    
    def show_instructions(self):
        """Mostra instruções de uso"""
        print("\n" + "="*70)
        print("🎯 CRIADOR DE MÚLTIPLAS ROIs")
        print("="*70)
        print("CONTROLES:")
        print("• Clique ESQUERDO: Adicionar ponto ao polígono atual")
        print("• Clique DIREITO: Finalizar polígono atual")
        print("• 'N': Iniciar novo polígono (finaliza o atual se tiver 3+ pontos)")
        print("• 'R': Resetar todos os polígonos")
        print("• 'U': Desfazer último polígono completo")
        print("• 'Enter': Salvar todas as ROIs e sair")
        print("• 'Esc': Cancelar")
        print("• 'H': Mostrar esta ajuda")
        print("="*70)
        print("DICA: Cada polígono terá uma cor diferente!")
        print("="*70)
    
    def save_multiple_rois(self, roi_polygons, save_path):
        """Salva múltiplos polígonos ROI"""
        try:
            save_path = Path(save_path)
            
            # Salva em formato pickle
            with open(save_path.with_suffix('.pkl'), 'wb') as f:
                pickle.dump(roi_polygons, f)
            
            # Salva também em JSON para visualização
            import json
            json_data = {
                'total_polygons': len(roi_polygons),
                'created_at': str(datetime.now()),
                'polygons': []
            }
            
            for i, poly in enumerate(roi_polygons):
                json_data['polygons'].append({
                    'id': i+1,
                    'points_count': len(poly),
                    'points': poly
                })
            
            with open(save_path.with_suffix('.json'), 'w') as f:
                json.dump(json_data, f, indent=2)
            
            print(f"✅ {len(roi_polygons)} ROIs salvas em: {save_path.with_suffix('.pkl')}")
            print(f"✅ JSON salvo em: {save_path.with_suffix('.json')}")
            
        except Exception as e:
            print(f"❌ Erro ao salvar ROIs: {e}")


def main():
    """Função principal para testar o criador de ROI"""
    
    # Configurações
    IMAGE_PATH = input("Digite o caminho da imagem (ou Enter para usar padrão): ").strip()
    if not IMAGE_PATH:
        IMAGE_PATH = r"D:\Sentric\Obra_Area\img\sample.webp"  # Altere para uma imagem sua
    
    ROI_SAVE_PATH = r"D:\Sentric\Obra_Area\multiple_rois"
    
    # Verifica se a imagem existe
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ Imagem não encontrada: {IMAGE_PATH}")
        return
    
    # Cria o criador de ROI
    roi_creator = MultiROICreator()
    
    # Cria as ROIs
    rois = roi_creator.create_multiple_rois(IMAGE_PATH, ROI_SAVE_PATH)
    
    if rois:
        print(f"\n✅ {len(rois)} ROIs criadas com sucesso!")
        for i, roi in enumerate(rois):
            print(f"   ROI {i+1}: {len(roi)} pontos")
    else:
        print("\n❌ Nenhuma ROI foi criada")


if __name__ == "__main__":
    main()