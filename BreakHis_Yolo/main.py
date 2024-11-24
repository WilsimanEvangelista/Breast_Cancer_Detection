from ultralytics import YOLO
from dotenv import dotenv_values

def main():
    env = dotenv_values("paths.env")

    model = YOLO("yolo8n-cls.pt")

    results = model.train(
        data=env["DATA_PATH"], 
        epochs=100,
        imgsz=224,
        batch=4,
        device='cuda',
        optimizer='SGD',
        patience=25)
    
if __name__ == "__main__":
    main()