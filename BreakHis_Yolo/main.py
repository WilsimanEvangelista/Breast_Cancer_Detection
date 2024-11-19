from ultralytics import YOLO
from dotenv import dotenv_values

env = dotenv_values("paths.env")

model = YOLO("yolo11n-cls.pt")

results = model.train(
    data=env["DATA"], 
    epochs=100,
    batch_size=4,
    imgsz=240,
    device='cuda',
    optimizer='SGD',
    patience=25)