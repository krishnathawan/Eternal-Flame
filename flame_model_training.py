from ultralytics import YOLO

# Load the pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')

# Train the model on your dataset
results = model.train(
    data=r"C:\Users\krish\candle.v1i.yolov8\data.yaml",  # Path to your dataset.yaml file
    epochs=50,                           # Number of training epochs
    batch =16,                       # Batch size
    imgsz=416                           # Image size (height and width)
)

# Save the trained model
model.save('trained_flame_detection_model.pt')


