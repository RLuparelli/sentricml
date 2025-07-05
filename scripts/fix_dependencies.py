#!/usr/bin/env python3
"""
Script para diagnosticar e corrigir problemas de dependências
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Executa comando e mostra resultado"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   ✅ {description} concluído")
            return True
        else:
            print(f"   ❌ Erro em {description}")
            print(f"   📄 Saída: {result.stderr}")
            return False
    except Exception as e:
        print(f"   ❌ Erro ao executar {description}: {e}")
        return False

def check_python_version():
    """Verifica versão do Python"""
    print("🐍 Verificando versão do Python...")
    version = sys.version_info
    print(f"   📊 Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("   ⚠️  Python 3.8+ é recomendado")
        return False
    else:
        print("   ✅ Versão do Python adequada")
        return True

def check_virtual_environment():
    """Verifica se está em ambiente virtual"""
    print("🏠 Verificando ambiente virtual...")
    
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("   ✅ Ambiente virtual ativo")
        return True
    else:
        print("   ⚠️  Não está em ambiente virtual")
        print("   💡 Recomendado usar ambiente virtual")
        return False

def upgrade_pip():
    """Atualiza pip"""
    print("📦 Atualizando pip...")
    return run_command("python -m pip install --upgrade pip", "Atualização do pip")

def install_basic_dependencies():
    """Instala dependências básicas"""
    print("🔧 Instalando dependências básicas...")
    
    basic_packages = [
        "wheel",
        "setuptools",
        "numpy",
        "pillow",
        "opencv-python"
    ]
    
    for package in basic_packages:
        if not run_command(f"pip install {package}", f"Instalação de {package}"):
            return False
    
    return True

def install_pytorch():
    """Instala PyTorch"""
    print("🔥 Instalando PyTorch...")
    
    # Verifica se CUDA está disponível
    try:
        import torch
        if torch.cuda.is_available():
            print("   🎯 CUDA detectado, instalando versão GPU")
            command = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
        else:
            print("   💻 CUDA não detectado, instalando versão CPU")
            command = "pip install torch torchvision torchaudio"
    except ImportError:
        print("   💻 PyTorch não encontrado, instalando versão CPU")
        command = "pip install torch torchvision torchaudio"
    
    return run_command(command, "Instalação do PyTorch")

def install_ultralytics():
    """Instala Ultralytics"""
    print("🎯 Instalando Ultralytics...")
    return run_command("pip install ultralytics", "Instalação do Ultralytics")

def install_additional_packages():
    """Instala pacotes adicionais"""
    print("📚 Instalando pacotes adicionais...")
    
    additional_packages = [
        "supervision",
        "scipy",
        "pandas",
        "matplotlib",
        "tqdm",
        "pathlib2"
    ]
    
    for package in additional_packages:
        run_command(f"pip install {package}", f"Instalação de {package}")
    
    return True

def test_imports():
    """Testa todas as importações"""
    print("🧪 Testando importações...")
    
    imports_to_test = [
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("PIL", "Pillow"),
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("ultralytics", "Ultralytics"),
        ("supervision", "Supervision"),
        ("scipy", "SciPy"),
        ("pandas", "Pandas"),
        ("matplotlib", "Matplotlib")
    ]
    
    all_passed = True
    
    for module, name in imports_to_test:
        try:
            __import__(module)
            print(f"   ✅ {name}")
        except ImportError as e:
            print(f"   ❌ {name}: {e}")
            all_passed = False
    
    return all_passed

def create_test_script():
    """Cria script de teste"""
    print("📝 Criando script de teste...")
    
    test_code = '''#!/usr/bin/env python3
"""
Script de teste das importações
"""

import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO

print("SUCESSO: Todas as importacoes funcionaram!")
print(f"OpenCV: {cv2.__version__}")
print(f"NumPy: {np.__version__}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA disponivel: {torch.cuda.is_available()}")

# Teste basico
img = np.zeros((100, 100, 3), dtype=np.uint8)
cv2.rectangle(img, (10, 10), (90, 90), (255, 255, 255), 2)
pil_img = Image.fromarray(img)
tensor = torch.tensor([1.0, 2.0, 3.0])

print("SUCESSO: Testes basicos passaram!")
print("Sistema pronto para uso!")
'''
    
    with open("test_imports.py", "w", encoding='utf-8') as f:
        f.write(test_code)
    
    print("   📄 Script de teste criado: test_imports.py")
    return True

def main():
    """Função principal"""
    
    print("🔧 CORREÇÃO DE DEPENDÊNCIAS")
    print("=" * 50)
    
    # Verifica Python
    if not check_python_version():
        print("❌ Versão do Python inadequada!")
        return False
    
    # Verifica ambiente virtual
    check_virtual_environment()
    
    # Atualiza pip
    upgrade_pip()
    
    # Instala dependências básicas
    if not install_basic_dependencies():
        print("❌ Falha na instalação de dependências básicas!")
        return False
    
    # Instala PyTorch
    install_pytorch()
    
    # Instala Ultralytics
    install_ultralytics()
    
    # Instala pacotes adicionais
    install_additional_packages()
    
    # Testa importações
    print("\n" + "=" * 50)
    if test_imports():
        print("✅ TODAS AS DEPENDÊNCIAS INSTALADAS COM SUCESSO!")
        
        # Cria script de teste
        create_test_script()
        
        print("\n🎯 Próximos passos:")
        print("   1. Execute 'python test_imports.py' para teste final")
        print("   2. Execute 'python quick_test.py' para teste completo")
        print("   3. Execute 'python take_single_picture.py' para começar")
        
        return True
    else:
        print("❌ ALGUMAS DEPENDÊNCIAS AINDA COM PROBLEMAS!")
        print("💡 Tente reinstalar manualmente os pacotes que falharam")
        return False

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n🎉 Correção concluída com sucesso!")
        else:
            print("\n💥 Correção falhou!")
    except KeyboardInterrupt:
        print("\n⚠️  Correção interrompida pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro inesperado: {e}")
        import traceback
        traceback.print_exc()
    
    input("\nPressione Enter para sair...")