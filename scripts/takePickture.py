import cv2
import os
from pathlib import Path

def extrair_frames_por_segundo(caminho_video, pasta_destino):
    """
    Extrai 1 frame por segundo de um vídeo e salva como imagens.
    
    Args:
        caminho_video (str): Caminho completo para o arquivo de vídeo
        pasta_destino (str): Caminho da pasta onde salvar as imagens
    """
    
    # Verificar se o arquivo de vídeo existe
    if not os.path.exists(caminho_video):
        print(f"Erro: O arquivo de vídeo '{caminho_video}' não foi encontrado.")
        return
    
    # Criar pasta de destino se não existir
    Path(pasta_destino).mkdir(parents=True, exist_ok=True)
    
    # Abrir o vídeo
    cap = cv2.VideoCapture(caminho_video)
    
    if not cap.isOpened():
        print(f"Erro: Não foi possível abrir o vídeo '{caminho_video}'.")
        return
    
    # Obter informações do vídeo
    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames por segundo
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duracao = total_frames / fps
    
    print(f"Informações do vídeo:")
    print(f"- FPS: {fps:.2f}")
    print(f"- Total de frames: {total_frames}")
    print(f"- Duração: {duracao:.2f} segundos")
    print(f"- Frames a serem extraídos: {int(duracao)}")
    
    # Nome base do arquivo (sem extensão)
    nome_video = Path(caminho_video).stem
    
    contador_frame = 0
    segundo_atual = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Calcular em que segundo estamos
        segundo_do_frame = contador_frame / fps
        
        # Se chegamos no próximo segundo, salvar o frame
        if int(segundo_do_frame) >= segundo_atual:
            nome_arquivo = f"{nome_video}_segundo_{segundo_atual:04d}.jpg"
            caminho_completo = os.path.join(pasta_destino, nome_arquivo)
            
            # Salvar o frame
            cv2.imwrite(caminho_completo, frame)
            print(f"Frame salvo: {nome_arquivo}")
            
            segundo_atual += 2
        
        contador_frame += 2
    
    cap.release()
    print(f"\nExtração concluída! {segundo_atual} frames salvos em '{pasta_destino}'")

# Configuração dos caminhos
caminho_video = r"D:\Sentric\MercadoLivre\dataset3.0\sentricml\videos\01976bb6-7ef2-71c3-9950-3ed41e504f94_2025-07-02_16-53-41-651590.mp4"
pasta_fotos = r"D:\Sentric\MercadoLivre\dataset3.0\sentricml\photos"

# Executar a extração
if __name__ == "__main__":
    extrair_frames_por_segundo(caminho_video, pasta_fotos)