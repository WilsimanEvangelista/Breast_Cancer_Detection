from k_fold_cross_validation.RepeatedStratifiedKFold_Classification import RepeatedStratifiedKFoldYOLO
from dotenv import dotenv_values
from ultralytics import YOLO
from types import NoneType
from pathlib import Path

class YOLOTrainer:
    def __init__(self, data_path, yolo_model, optimizer, batch_size, epochs, patience, seed, project_name):
        self.data_path = data_path
        self.yolo_model = yolo_model
        self.optimizer = optimizer
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.patience = int(patience)
        self.seed = int(seed)
        self.project_name = project_name

    def train(self, data_path):
        model = YOLO(f'{self.yolo_model}.pt')
        val_results = model.train(
            data=data_path,
            epochs=self.epochs,
            imgsz=224,
            batch=self.batch_size,
            patience=self.patience,
            device='cuda',
            optimizer=self.optimizer,
            seed=self.seed,
            project=self.project_name,
            name="train"
        )
        return model, val_results

    def test_yolo_model(self, model, data_path:str) -> list:
        try:
            metrics = model.val(
                data=data_path,
                split="test",
                task="classify",
                plots=True,
                save_json=True,
                project=self.project_name,
                name="test"
            )
            return metrics
        
        except Exception as e:
            print(f"Não foi possível fazer o teste do modelo.\n\n\n ERRO:\n\n\n{e}")

if __name__ == "__main__":
    env = dotenv_values("paths.env")
    
    yolo_trainer = YOLOTrainer(
        data_path=Path(env["DATA_PATH"]),
        yolo_model=env["YOLO_MODEL"],
        optimizer=env["OPTIMIZER"],
        batch_size=env["BATCH_SIZE"],
        epochs=env["EPOCHS"],
        patience=env["PATIENCE"],
        seed=env["SEED"],
        project_name=env["PROJECT_NAME"]
    )

    yolo_trainer.train_path = yolo_trainer.data_path / "train"
    yolo_trainer.val_path = yolo_trainer.data_path / "val"
    yolo_trainer.test_path = yolo_trainer.data_path / "test"
    
    if (env["CROSS_VALIDATION"]).upper() == "REPEATEDSTRATIFIEDKFOLD":
        cross_validator = RepeatedStratifiedKFoldYOLO(
            n_splits=env["N_SPLITS"],
            n_repeats=env["N_REPEATS"],
            random_state=env["SEED"],
            csv_path=env["CSV_PATH"]
        )
        results = cross_validator.run(yolo_trainer)
        print(results)