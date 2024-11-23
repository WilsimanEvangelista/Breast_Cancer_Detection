import os
import shutil

def move_images_from_txt(txt_file, destination_folder):
    # Certifique-se de que a pasta de destino exista
    os.makedirs(destination_folder, exist_ok=True)

    # Leia os caminhos das imagens no arquivo .txt
    with open(txt_file, 'r') as file:
        image_paths = file.readlines()

    for image_path in image_paths:
        # Remove espaços extras ou quebras de linha
        image_path = f"C:\\Users\\Wilsiman Evangelista\\Desktop\\Breast_Cancer_Detection\\BreakHis_Yolo\\40X_multiclass\\{(image_path.strip().split("/"))[1]}\\{(image_path.strip().split("/"))[2]}"

        if os.path.isfile(image_path):  # Verifica se o arquivo existe
            try:
                # Move o arquivo para a pasta de destino
                shutil.move(image_path, destination_folder)
                print(f"Imagem movida: {image_path}")
            except Exception as e:
                print(f"Erro ao mover {image_path}: {e}")
        else:
            print(f"Arquivo não encontrado: {image_path}")

# Uso:
txt_file = "C:\\Users\\Wilsiman Evangelista\\Desktop\\Breast_Cancer_Detection\\BreakHis_Yolo\\40X_multiclass\\tubular_adenoma_autosplit_test.txt"  # Arquivo .txt com os caminhos
destination_folder = "C:\\Users\\Wilsiman Evangelista\\Desktop\\Breast_Cancer_Detection\\BreakHis_Yolo\\multiclass_dataset\\test\\tubular_adenoma"  # Pasta de destino

move_images_from_txt(txt_file, destination_folder)