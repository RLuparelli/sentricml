# Configuração do Ambiente Virtual Python

Este documento descreve como configurar um ambiente virtual Python e instalar as dependências listadas no arquivo `requirements.txt`.

## Pré-requisitos

- Python 3.8 ou superior instalado. Verifique com:
  ```bash
  python3 --version
  ```
- `pip` (gerenciador de pacotes Python) instalado. Verifique com:
  ```bash
  pip3 --version
  ```

## Passos para Configuração

1. **Criar um ambiente virtual**  
   No diretório do projeto, execute o comando abaixo para criar um ambiente virtual chamado `venv`:
   ```bash
   python3 -m venv venv
   ```

2. **Ativar o ambiente virtual**  

   - **Windows**:
     ```bash
     venv\Scripts\activate
     ```

   Após a ativação, você verá `(venv)` no início do prompt do terminal.

3. **Instalar dependências**  
   Com o ambiente virtual ativado, instale as dependências listadas no arquivo `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

4. ## Sair do Ambiente Virtual

Para desativar o ambiente virtual e retornar ao ambiente global do Python, execute:

```bash
deactivate   
