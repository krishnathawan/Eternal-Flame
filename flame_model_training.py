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

# Validate the trained model on the test set
metrics = model.val()
# Access the results dictionary
results = metrics.results_dict

# Extract precision, recall, mAP50, and mAP50-95 values
precision = results['metrics/precision(B)']  # Precision
recall = results['metrics/recall(B)']          # Recall
mAP50 = results['metrics/mAP50(B)']            # mAP at IoU=0.50
mAP50_95 = results['metrics/mAP50-95(B)']      # mAP at IoU=0.50 to 0.95

print(f'Precision: {precision:.3f}, Recall: {recall:.3f}, mAP50: {mAP50:.3f}, mAP50-95: {mAP50_95:.3f}')


