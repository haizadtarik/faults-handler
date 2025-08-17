from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-seg.pt")  # load a pretrained model (recommended for training)
# Train the model
results = model.train(data="config/seimic_data.yaml", epochs=3, imgsz=768, batch=8, show_boxes=False)