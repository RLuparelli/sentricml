import os
import cv2
from pathlib import Path
from ultralytics import YOLO
import numpy as np

def processar_fotos_com_yolo(pasta_fotos, modelo_path, pasta_destino, confianca_min=0.1):
    """
    Processa todas as fotos de uma pasta usando modelo YOLO e salva os resultados.
    
    Args:
        pasta_fotos (str): Pasta com as fotos originais
        modelo_path (str): Caminho para o modelo YOLO (.pt)
        pasta_destino (str): Pasta onde salvar as fotos processadas
        confianca_min (float): Confiança mínima para detecções (0.0 a 1.0)
    """
    
    # Verificar se o modelo existe
    if not os.path.exists(modelo_path):
        print(f"Erro: Modelo YOLO '{modelo_path}' não encontrado.")
        return
    
    # Verificar se a pasta de fotos existe
    if not os.path.exists(pasta_fotos):
        print(f"Erro: Pasta de fotos '{pasta_fotos}' não encontrada.")
        return
    
    # Criar pasta de destino se não existir
    Path(pasta_destino).mkdir(parents=True, exist_ok=True)
    
    # Carregar o modelo YOLO
    print(f"Carregando modelo YOLO: {modelo_path}")
    try:
        modelo = YOLO(modelo_path)
        print("Modelo carregado com sucesso!")
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        return
    
    # Extensões de imagem suportadas
    extensoes_validas = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # Buscar todas as imagens na pasta
    fotos = []
    for arquivo in os.listdir(pasta_fotos):
        if Path(arquivo).suffix.lower() in extensoes_validas:
            fotos.append(arquivo)
    
    if not fotos:
        print(f"Nenhuma imagem encontrada na pasta '{pasta_fotos}'")
        return
    
    print(f"Encontradas {len(fotos)} imagens para processar")
    print(f"Confiança mínima: {confianca_min}")
    print("-" * 50)
    
    # Processar cada foto
    fotos_processadas = 0
    total_deteccoes = 0
    
    for i, nome_foto in enumerate(fotos, 1):
        caminho_foto = os.path.join(pasta_fotos, nome_foto)
        
        try:
            # Carregar a imagem
            imagem = cv2.imread(caminho_foto)
            if imagem is None:
                print(f"Erro: Não foi possível carregar '{nome_foto}'")
                continue
            
            # Fazer a inferência
            resultados = modelo(imagem, conf=confianca_min)
            
            # Desenhar as detecções na imagem
            imagem_anotada = imagem.copy()
            deteccoes_foto = 0
            
            for resultado in resultados:
                boxes = resultado.boxes
                if boxes is not None:
                    for box in boxes:
                        # Coordenadas da caixa
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        confianca = box.conf[0].cpu().numpy()
                        classe_id = int(box.cls[0].cpu().numpy())
                        
                        # Nome da classe (se disponível)
                        if hasattr(modelo, 'names') and classe_id < len(modelo.names):
                            nome_classe = modelo.names[classe_id]
                        else:
                            nome_classe = f"Classe_{classe_id}"
                        
                        # Desenhar retângulo
                        cv2.rectangle(imagem_anotada, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Texto da detecção
                        texto = f"{nome_classe}: {confianca:.2f}"
                        tamanho_texto = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        
                        # Fundo do texto
                        cv2.rectangle(imagem_anotada, 
                                    (x1, y1 - tamanho_texto[1] - 10), 
                                    (x1 + tamanho_texto[0], y1), 
                                    (0, 255, 0), -1)
                        
                        # Texto
                        cv2.putText(imagem_anotada, texto, (x1, y1 - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                        
                        deteccoes_foto += 1
            
            # Salvar a imagem processada
            nome_base = Path(nome_foto).stem
            extensao = Path(nome_foto).suffix
            nome_processado = f"{nome_base}_processado{extensao}"
            caminho_destino = os.path.join(pasta_destino, nome_processado)
            
            cv2.imwrite(caminho_destino, imagem_anotada)
            
            print(f"[{i}/{len(fotos)}] {nome_foto} -> {deteccoes_foto} detecções")
            
            fotos_processadas += 1
            total_deteccoes += deteccoes_foto
            
        except Exception as e:
            print(f"Erro ao processar '{nome_foto}': {e}")
            continue
    
    print("-" * 50)
    print(f"Processamento concluído!")
    print(f"- Fotos processadas: {fotos_processadas}/{len(fotos)}")
    print(f"- Total de detecções: {total_deteccoes}")
    print(f"- Fotos salvas em: {pasta_destino}")

def listar_classes_modelo(modelo_path):
    """Lista as classes que o modelo foi treinado para detectar."""
    try:
        modelo = YOLO(modelo_path)
        if hasattr(modelo, 'names'):
            print("Classes do modelo:")
            for i, nome in modelo.names.items():
                print(f"  {i}: {nome}")
        else:
            print("Informações de classes não disponíveis no modelo.")
    except Exception as e:
        print(f"Erro ao carregar modelo: {e}")

# Configuração dos caminhos
pasta_fotos = r"C:\Users\lupar\Desktop\Sentric\MercadoLivre\Scripts\fotos2"
modelo_yolo = r"C:\Users\lupar\Desktop\Sentric\MercadoLivre\Scripts\MercadoLivreBest.pt"
pasta_processadas = r"C:\Users\lupar\Desktop\Sentric\MercadoLivre\Scripts\fotosProcessadas2"

# Configurações de inferência
confianca_minima = 0.5  # Ajuste conforme necessário (0.1 a 0.9)

if __name__ == "__main__":
    print("=== SCRIPT DE INFERÊNCIA YOLO ===")
    print()
    
    # Mostrar classes do modelo (opcional)
    print("Verificando classes do modelo...")
    listar_classes_modelo(modelo_yolo)
    print()
    
    # Processar as fotos
    processar_fotos_com_yolo(
        pasta_fotos=pasta_fotos,
        modelo_path=modelo_yolo,
        pasta_destino=pasta_processadas,
        confianca_min=confianca_minima
    )