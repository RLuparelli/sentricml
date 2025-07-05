import cv2
import os
from pathlib import Path
from datetime import datetime

def capturar_foto_unica(caminho_video, pasta_destino, segundo_desejado=0):
    """
    Captura uma única foto de um vídeo no segundo especificado.
    
    Args:
        caminho_video (str): Caminho completo para o arquivo de vídeo
        pasta_destino (str): Caminho da pasta onde salvar a foto
        segundo_desejado (int): Segundo do vídeo para capturar (padrão: 0)
    
    Returns:
        str: Caminho da foto capturada ou None se houver erro
    """
    
    # Verificar se o arquivo de vídeo existe
    if not os.path.exists(caminho_video):
        print(f"❌ Erro: O arquivo de vídeo '{caminho_video}' não foi encontrado.")
        return None
    
    # Criar pasta de destino se não existir
    Path(pasta_destino).mkdir(parents=True, exist_ok=True)
    
    # Abrir o vídeo
    cap = cv2.VideoCapture(caminho_video)
    
    if not cap.isOpened():
        print(f"❌ Erro: Não foi possível abrir o vídeo '{caminho_video}'.")
        return None
    
    # Obter informações do vídeo
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duracao = total_frames / fps
    
    print(f"📹 Informações do vídeo:")
    print(f"   - FPS: {fps:.2f}")
    print(f"   - Total de frames: {total_frames}")
    print(f"   - Duração: {duracao:.2f} segundos")
    
    # Verificar se o segundo desejado está dentro da duração
    if segundo_desejado >= duracao:
        print(f"⚠️  Aviso: Segundo desejado ({segundo_desejado}) está além da duração do vídeo ({duracao:.2f}s)")
        segundo_desejado = int(duracao / 2)  # Usar o meio do vídeo
        print(f"   Usando segundo {segundo_desejado} (meio do vídeo)")
    
    # Calcular o frame desejado
    frame_desejado = int(segundo_desejado * fps)
    
    # Posicionar no frame desejado
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_desejado)
    
    # Capturar o frame
    ret, frame = cap.read()
    
    if not ret:
        print(f"❌ Erro: Não foi possível capturar o frame no segundo {segundo_desejado}")
        cap.release()
        return None
    
    # Gerar nome único para a foto
    nome_video = Path(caminho_video).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nome_foto = f"{nome_video}_frame_s{segundo_desejado:04d}_{timestamp}.jpg"
    caminho_foto = os.path.join(pasta_destino, nome_foto)
    
    # Salvar a foto
    sucesso = cv2.imwrite(caminho_foto, frame)
    
    cap.release()
    
    if sucesso:
        print(f"✅ Foto capturada com sucesso!")
        print(f"   - Segundo: {segundo_desejado}")
        print(f"   - Frame: {frame_desejado}")
        print(f"   - Salva em: {caminho_foto}")
        return caminho_foto
    else:
        print(f"❌ Erro ao salvar a foto em '{caminho_foto}'")
        return None

def capturar_foto_interativa(caminho_video, pasta_destino):
    """
    Modo interativo para escolher o frame a ser capturado.
    
    Args:
        caminho_video (str): Caminho para o vídeo
        pasta_destino (str): Pasta de destino
    
    Returns:
        str: Caminho da foto capturada ou None
    """
    
    if not os.path.exists(caminho_video):
        print(f"❌ Vídeo não encontrado: {caminho_video}")
        return None
    
    cap = cv2.VideoCapture(caminho_video)
    
    if not cap.isOpened():
        print(f"❌ Erro ao abrir vídeo")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duracao = total_frames / fps
    
    print(f"📹 Vídeo carregado - Duração: {duracao:.2f}s")
    print(f"🎮 Controles:")
    print(f"   - Setas ←→: Navegar pelos frames")
    print(f"   - Espaço: Capturar foto atual")
    print(f"   - ESC: Sair sem capturar")
    
    frame_atual = 0
    janela_nome = "Selecionador de Frame - Pressione ESPAÇO para capturar"
    
    cv2.namedWindow(janela_nome, cv2.WINDOW_AUTOSIZE)
    
    while True:
        # Posicionar no frame atual
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_atual)
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Adicionar informações no frame
        segundo_atual = frame_atual / fps
        info_text = f"Frame: {frame_atual}/{total_frames} | Segundo: {segundo_atual:.2f}s"
        
        frame_info = frame.copy()
        cv2.putText(frame_info, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame_info, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
        
        cv2.imshow(janela_nome, frame_info)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            print("❌ Captura cancelada")
            break
        elif key == 32:  # Espaço
            # Capturar foto atual
            nome_video = Path(caminho_video).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            nome_foto = f"{nome_video}_frame_{frame_atual:06d}_{timestamp}.jpg"
            caminho_foto = os.path.join(pasta_destino, nome_foto)
            
            Path(pasta_destino).mkdir(parents=True, exist_ok=True)
            
            if cv2.imwrite(caminho_foto, frame):
                print(f"✅ Foto capturada!")
                print(f"   - Frame: {frame_atual}")
                print(f"   - Segundo: {segundo_atual:.2f}s")
                print(f"   - Salva em: {caminho_foto}")
                cap.release()
                cv2.destroyAllWindows()
                return caminho_foto
            else:
                print(f"❌ Erro ao salvar foto")
        elif key == 81:  # Seta esquerda
            frame_atual = max(0, frame_atual - int(fps))  # Voltar 1 segundo
        elif key == 83:  # Seta direita
            frame_atual = min(total_frames - 1, frame_atual + int(fps))  # Avançar 1 segundo
        elif key == 82:  # Seta para cima
            frame_atual = min(total_frames - 1, frame_atual + int(fps * 10))  # Avançar 10 segundos
        elif key == 84:  # Seta para baixo
            frame_atual = max(0, frame_atual - int(fps * 10))  # Voltar 10 segundos
    
    cap.release()
    cv2.destroyAllWindows()
    return None

def main():
    """Função principal com menu de opções"""
    
    print("=" * 60)
    print("🎬 CAPTURADOR DE FOTO ÚNICA DE VÍDEO")
    print("=" * 60)
    
    # Configurações padrão
    caminho_video_padrao = r"D:\Sentric\MercadoLivre\dataset3.0\sentricml\videos"
    pasta_fotos_padrao = r"D:\Sentric\MercadoLivre\dataset3.0\sentricml\photos"
    
    print(f"📁 Pasta de vídeos padrão: {caminho_video_padrao}")
    print(f"📁 Pasta de fotos padrão: {pasta_fotos_padrao}")
    print()
    
    # Listar vídeos disponíveis
    if os.path.exists(caminho_video_padrao):
        videos = [f for f in os.listdir(caminho_video_padrao) 
                 if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        if videos:
            print("🎥 Vídeos disponíveis:")
            for i, video in enumerate(videos, 1):
                print(f"   {i}. {video}")
            print()
        else:
            print("⚠️  Nenhum vídeo encontrado na pasta padrão")
    
    # Escolher vídeo
    caminho_video = input("📹 Digite o caminho completo do vídeo (ou Enter para usar padrão): ").strip()
    
    if not caminho_video:
        if 'videos' in locals() and videos:
            try:
                escolha = int(input(f"Escolha um vídeo (1-{len(videos)}): ")) - 1
                if 0 <= escolha < len(videos):
                    caminho_video = os.path.join(caminho_video_padrao, videos[escolha])
                else:
                    print("❌ Escolha inválida")
                    return
            except ValueError:
                print("❌ Entrada inválida")
                return
        else:
            print("❌ Nenhum vídeo disponível")
            return
    
    # Escolher pasta de destino
    pasta_destino = input("📁 Pasta de destino (Enter para usar padrão): ").strip()
    if not pasta_destino:
        pasta_destino = pasta_fotos_padrao
    
    # Escolher modo
    print("\n🎯 Escolha o modo:")
    print("1. Capturar frame específico por segundo")
    print("2. Modo interativo (navegar e escolher)")
    
    try:
        modo = int(input("Modo (1 ou 2): "))
        
        if modo == 1:
            segundo = int(input("⏱️  Segundo para capturar (0 para início): "))
            foto_path = capturar_foto_unica(caminho_video, pasta_destino, segundo)
        elif modo == 2:
            foto_path = capturar_foto_interativa(caminho_video, pasta_destino)
        else:
            print("❌ Modo inválido")
            return
        
        if foto_path:
            print(f"\n🎉 Sucesso! Foto salva em: {foto_path}")
            print(f"📏 Use esta foto para criar a ROI no próximo script")
        else:
            print("\n❌ Nenhuma foto foi capturada")
            
    except ValueError:
        print("❌ Entrada inválida")
    except KeyboardInterrupt:
        print("\n⚠️  Operação cancelada pelo usuário")

if __name__ == "__main__":
    main()