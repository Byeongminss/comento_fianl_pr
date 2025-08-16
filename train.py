from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model.train(data="datasets/data.yaml", epochs=20, imgsz=640)

print("Training complete!")