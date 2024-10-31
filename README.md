**Eternal Flame Detection Experiment**
This experiment leverages transfer learning with the YOLOv8n model to monitor an eternal flame. By segmenting images to isolate and measure flame height, it provides real-time information when the flame dims .

**Why YOLO?**
YOLO (You Only Look Once) is a state-of-the-art deep learning model known for its speed and accuracy in object detection. It simultaneously predicts bounding boxes and class probabilities in real time, making it highly efficient for applications that require fast response times. YOLO was selected for this experiment due to:

(1) Real-Time Detection: YOLO's architecture allows it to process frames quickly, making it ideal for live flame monitoring.
(2) Transfer Learning: Using YOLOv8n’s pre-trained weights, we were able to fine-tune the model to detect and measure flame height with high precision, saving training time and computational resources.
(3) High Accuracy: YOLO's reliability in detecting small, defined objects like flames helps ensure that only actual flame changes (like dimming or extinguishing) trigger alerts.
Methodology
(4) Transfer Learning with YOLOv8n: Fine-tuned the YOLOv8n model specifically for flame detection.
(5) Image Segmentation: Used to isolate and measure flame height, enabling accurate monitoring and alerting.

**Dataset and Annotation**: Pre-annotated data was sourced from Roboflow. Tools like bbox can also be used, but pre-annotated data was preferred for efficiency.

**Implementation Flow**
(1) Define .yaml File: Specifies the annotated dataset for training.
(2) Train the Model: Run flame_model_training.py with the .yaml path to train and save the model.
(3) Live Monitoring: Load the trained model with live_flame_detection.py for real-time monitoring.

**Real-World Applications**
(1) Fire Safety: Suitable for settings requiring constant flame monitoring, like religious or ceremonial sites.
(2) Advantages Over Fire Sensors: Using a camera allows for direct visual confirmation and precise flame height measurement, which reduces false alarms and enhances reliability.

**Real-Life Implementation**
A Raspberry Pi can host the trained model and connect to a camera for fully autonomous flame monitoring in practical, real-world settings.
