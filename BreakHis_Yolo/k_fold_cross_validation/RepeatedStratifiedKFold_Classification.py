from sklearn.model_selection import RepeatedStratifiedKFold
from pathlib import Path
import pandas as pd
import numpy as np
import shutil
import os


class RepeatedStratifiedKFoldYOLO:
    def __init__(self, n_splits:int, n_repeats:int, random_state:int, csv_path:str):
        self.n_splits = int(n_splits)
        self.n_repeats = int(n_repeats)
        self.random_state = int(random_state)
        self.csv_path = str(csv_path)

    def get_labels(self, path):
        """Obtém os rótulos com base na estrutura de pastas."""
        class_dirs = [p for p in path.iterdir() if p.is_dir()]
        labels = []
        for class_idx, class_dir in enumerate(sorted(class_dirs)):
            for _ in class_dir.glob('*.png'):
                labels.append(class_idx)
        return labels

    def stratify_labels(self, labels):
        """Transforma os rótulos em uma forma adequada para estratificação."""
        return labels  # Labels já são categóricas para classificação.

    def run(self, yolo_trainer):
        train_path = yolo_trainer.train_path
        labels = self.get_labels(train_path)

        rskf = RepeatedStratifiedKFold(
            n_splits=self.n_splits,
            n_repeats=self.n_repeats,
            random_state=self.random_state
        )

        # Resultados
        try:
            df_resultados = pd.read_csv(self.csv_path)
        except:
            df_resultados = pd.DataFrame(columns=["fold", "pretrained_model", "batchsize", "optimizer", 
                                                  "val_top1_acc", "val_top5_acc","test_top1_acc", "test_top5_acc"])

        image_files = sorted(train_path.rglob("*.png"))
        for fold, (train_idx, val_idx) in enumerate(rskf.split(image_files, labels), 1):
            print(f"\n\n Treinando Fold {fold} \n\n")
            
            if len(df_resultados.loc[(df_resultados.loc[:, "fold"] == fold) & (df_resultados.loc[:, "pretrained_model"] == yolo_trainer.yolo_model) & (df_resultados.loc[:, "batchsize"] == yolo_trainer.batch_size) & (df_resultados.loc[:, "optimizer"] == yolo_trainer.optimizer), :]) > 0:
                continue

            # Criar estrutura para o fold
            fold_train, fold_val = self.create_fold_directories(yolo_trainer, fold, train_idx, val_idx, image_files)

            try:
                # Treinamento
                model, val_results = yolo_trainer.train(str(fold_train))

                # Teste
                test_results = yolo_trainer.test_yolo_model(model, yolo_trainer.data_path)

                # Atualizar resultados
                df_resultados = self.update_results(df_resultados, fold, yolo_trainer, val_results, test_results)
            
            except Exception as e:
                print(f"Exception in fold {fold}:\n{e}")
                df_resultados = self.update_results_with_nan(df_resultados, fold, yolo_trainer)

            # Salvar resultados
            df_resultados.to_csv(self.csv_path, index=False)

            self.delete_folder(str(fold_train))

        return df_resultados

    def create_fold_directories(self, yolo_trainer, fold, train_idx, val_idx, image_files):
        fold_train = yolo_trainer.data_path / f"fold_{fold}_train" / "train"
        fold_val = Path((str(fold_train).split("\\"))[0]) / Path((str(fold_train).split("\\"))[1]) / "val"

        os.makedirs(fold_train, exist_ok=True)
        os.makedirs(fold_val, exist_ok=True)

        for idx in train_idx:
            src = image_files[idx]
            dst = fold_train / src.relative_to(src.parent.parent)
            os.makedirs(dst.parent, exist_ok=True)
            shutil.copy(src, dst)

        for idx in val_idx:
            src = image_files[idx]
            dst = fold_val / src.relative_to(src.parent.parent)
            os.makedirs(dst.parent, exist_ok=True)
            shutil.copy(src, dst)

        return yolo_trainer.data_path / f"fold_{fold}_train", fold_val
    
    def delete_folder(self,folder_path):
        """
        Apaga uma pasta e todo o seu conteúdo, incluindo subpastas e arquivos.

        Args:
            caminho_pasta (str): Caminho absoluto ou relativo da pasta a ser apagada.

        Returns:
            bool: True se a pasta foi apagada com sucesso, False se a pasta não existia.
        """
        if os.path.exists(folder_path):
            try:
                shutil.rmtree(folder_path)
                return True
            except Exception as e:
                print(f"Erro ao apagar a pasta '{folder_path}': {e}")
                return False
        else:
            print(f"A pasta '{folder_path}' não existe.")
            return False

    def update_results(self, df_resultados, fold, yolo_trainer, val_results, test_results):
        return pd.concat([df_resultados, pd.DataFrame({
            "fold": [fold],
            "pretrained_model": [yolo_trainer.yolo_model],
            "batchsize": [yolo_trainer.batch_size],
            "optimizer": [yolo_trainer.optimizer],
            "val_top1_acc": [val_results.top1],
            "val_top5_acc": [val_results.top5],
            "test_top1_acc": [test_results.top1],
            "test_top5_acc": [test_results.top5],
        })], ignore_index=True)
    
    def update_results_with_nan(self, df_resultados, fold, yolo_trainer):
        return pd.concat([df_resultados, pd.DataFrame({
            "fold": [fold],
            "pretrained_model": [yolo_trainer.yolo_model],
            "batchsize": [yolo_trainer.batch_size],
            "optimizer": [yolo_trainer.optimizer],
            "val_top1_acc": [np.nan],
            "val_top5_acc": [np.nan],
            "test_top1_acc": [np.nan],
            "test_top5_acc": [np.nan],
        })], ignore_index=True)