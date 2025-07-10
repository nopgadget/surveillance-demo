from ultralytics import YOLO

model = YOLO("models/yolo11n.pt")

model.export(format="onnx")