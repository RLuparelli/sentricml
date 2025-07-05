#!/usr/bin/env python3
"""
Exemplo de uso do sistema de detecção de pacotes
"""

from package_detector_tracker import PackageDetector
from pathlib import Path

def main():
    # Caminhos
    model_path = Path(r"D:\Sentric\MercadoLivre\dataset3.0\sentricml\models\MercadoLivreBest.pt")
    video_path = Path(r"D:\Sentric\MercadoLivre\dataset3.0\sentricmlideos\seu_video.mp4")
    roi_path = Path(r"D:\Sentric\MercadoLivre\dataset3.0\sentricmloioi_config.json")
    output_path = Path(r"D:\Sentric\MercadoLivre\dataset3.0\sentricml\outputesultado.mp4")
    
    # Criar detector
    detector = PackageDetector(model_path, roi_path)
    
    # Processar vídeo
    results = detector.process_video(video_path, output_path)
    
    print(f"Pacotes detectados: {results['total_packages']}")

if __name__ == "__main__":
    main()
