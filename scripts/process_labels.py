import os
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Processa arquivos de labels YOLO para excluir classe 8 (label), mesclar outras classes em 0 e converter anotações com mais de 4 pontos para bounding boxes.')
parser.add_argument('dir_path', help='Caminho da pasta contendo os arquivos de labels')
args = parser.parse_args()
dir_path = args.dir_path

def convert_to_bbox(coords):
    if len(coords) <= 8:
        return coords
    coords = np.array(coords, dtype=float).reshape(-1, 2)
    x_min = coords[:, 0].min()
    x_max = coords[:, 0].max()
    y_min = coords[:, 1].min()
    y_max = coords[:, 1].max()
    return [x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]

if not os.path.isdir(dir_path):
    print(f"Erro: O diretório {dir_path} não existe.")
    exit(1)

for filename in os.listdir(dir_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(dir_path, filename)
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Erro ao ler o arquivo {file_path}: {e}")
            continue
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                class_id = int(parts[0])
                if class_id == 8:
                    continue
                if class_id in (0, 1, 2, 3, 4, 5, 6, 7, 9, 10):
                    coords = [float(x) for x in parts[1:]]
                    coords = convert_to_bbox(coords)
                    new_line = f"0 " + ' '.join(f"{x:.6f}" for x in coords) + '\n'
                    new_lines.append(new_line)
            except (ValueError, IndexError) as e:
                print(f"Erro ao processar linha em {file_path}: {line.strip()} - {e}")
                continue
        try:
            with open(file_path, 'w') as f:
                f.writelines(new_lines)
        except Exception as e:
            print(f"Erro ao escrever no arquivo {file_path}: {e}")
            continue