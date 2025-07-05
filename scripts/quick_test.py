#!/usr/bin/env python3
"""
Teste simples das dependências do sistema
"""

def test_all_imports():
    """Testa todas as importações necessárias"""
    
    print("=" * 60)
    print("TESTE DE DEPENDENCIAS DO SISTEMA")
    print("=" * 60)
    
    # Lista de testes
    tests = [
        ("OpenCV", "cv2"),
        ("NumPy", "numpy", "np"),
        ("Pillow", "PIL", "Image"),
        ("PyTorch", "torch"),
        ("TorchVision", "torchvision"),
        ("Ultralytics", "ultralytics", "YOLO"),
        ("Supervision", "supervision", "sv"),
        ("SciPy", "scipy"),
        ("Pandas", "pandas", "pd"),
        ("Matplotlib", "matplotlib", "plt"),
        ("JSON", "json"),
        ("OS", "os"),
        ("Pathlib", "pathlib", "Path"),
        ("Time", "time"),
        ("Collections", "collections")
    ]
    
    passed = 0
    failed = 0
    
    for test_info in tests:
        name = test_info[0]
        module = test_info[1]
        
        try:
            if len(test_info) == 3:
                # Importação específica
                exec(f"from {module} import {test_info[2]}")
            else:
                # Importação do módulo completo
                exec(f"import {module}")
            
            print(f"SUCESSO: {name}")
            passed += 1
        except ImportError as e:
            print(f"FALHA: {name} - {e}")
            failed += 1
        except Exception as e:
            print(f"ERRO: {name} - {e}")
            failed += 1
    
    print("=" * 60)
    print(f"RESULTADOS: {passed} sucessos, {failed} falhas")
    print("=" * 60)
    
    return failed == 0

def test_basic_functionality():
    """Testa funcionalidades básicas"""
    
    print("\nTESTE DE FUNCIONALIDADES BASICAS")
    print("-" * 60)
    
    try:
        # Teste OpenCV + NumPy
        import cv2
        import numpy as np
        
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(img, (10, 10), (90, 90), (255, 255, 255), 2)
        print(f"SUCESSO: OpenCV + NumPy - Imagem criada {img.shape}")
        
        # Teste PyTorch
        import torch
        tensor = torch.tensor([1.0, 2.0, 3.0])
        print(f"SUCESSO: PyTorch - Tensor criado {tensor.shape}")
        
        cuda_available = torch.cuda.is_available()
        print(f"INFO: CUDA disponivel: {cuda_available}")
        
        # Teste Pillow
        from PIL import Image
        pil_img = Image.new('RGB', (100, 100), color='red')
        print(f"SUCESSO: Pillow - Imagem criada {pil_img.size}")
        
        # Teste Ultralytics (sem modelo)
        from ultralytics import YOLO
        print("SUCESSO: Ultralytics importado")
        
        return True
        
    except Exception as e:
        print(f"ERRO: Teste de funcionalidades - {e}")
        return False

def check_model_file():
    """Verifica se o arquivo do modelo existe"""
    
    print("\nVERIFICACAO DO MODELO YOLO")
    print("-" * 60)
    
    from pathlib import Path
    
    model_path = Path(r"D:\Sentric\MercadoLivre\dataset3.0\sentricml\models\MercadoLivreBest.pt")
    
    if model_path.exists():
        print(f"SUCESSO: Modelo encontrado - {model_path}")
        
        # Tenta carregar o modelo
        try:
            from ultralytics import YOLO
            model = YOLO(str(model_path))
            print("SUCESSO: Modelo carregado com sucesso")
            
            if hasattr(model, 'names'):
                classes = list(model.names.values())
                print(f"INFO: Classes detectaveis: {classes}")
            
            return True
            
        except Exception as e:
            print(f"ERRO: Nao foi possivel carregar o modelo - {e}")
            return False
    else:
        print(f"AVISO: Modelo nao encontrado - {model_path}")
        print("INFO: Coloque o arquivo MercadoLivreBest.pt na pasta models")
        return False

def main():
    """Função principal"""
    
    # Teste de importações
    imports_ok = test_all_imports()
    
    # Teste de funcionalidades
    functionality_ok = test_basic_functionality()
    
    # Verificação do modelo
    model_ok = check_model_file()
    
    # Resultado final
    print("\n" + "=" * 60)
    print("RESULTADO FINAL")
    print("=" * 60)
    
    if imports_ok and functionality_ok:
        print("STATUS: SISTEMA PRONTO PARA USO!")
        print("\nPROXIMOS PASSOS:")
        print("1. Execute: python take_single_picture.py")
        print("2. Execute: python roi_creator_improved.py")
        print("3. Execute: python package_detector_tracker.py")
        
        if model_ok:
            print("\nINFO: Modelo YOLO funcionando perfeitamente!")
        else:
            print("\nAVISO: Verifique o arquivo do modelo YOLO")
            
    else:
        print("STATUS: SISTEMA COM PROBLEMAS!")
        if not imports_ok:
            print("ERRO: Problemas com importacoes")
        if not functionality_ok:
            print("ERRO: Problemas com funcionalidades")
    
    print("=" * 60)
    
    return imports_ok and functionality_ok

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\nCONCLUSAO: Teste passou com sucesso!")
        else:
            print("\nCONCLUSAO: Teste falhou!")
    except KeyboardInterrupt:
        print("\nINTERRUPCAO: Teste cancelado pelo usuario")
    except Exception as e:
        print(f"\nERRO INESPERADO: {e}")
        import traceback
        traceback.print_exc()
    
    input("\nPressione Enter para sair...")