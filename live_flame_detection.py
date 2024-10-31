#import needed libraries 
import cv2  
from ultralytics import YOLO # we will use pretrained yolo model , hence we imported it using ultralytics  
import winsound  # importing this library for beep sound on Windows

model = YOLO('trained_flame_detection_model.pt') # Load the trained YOLOv8 model 

# write camera index as input argument for live capturing
cap = cv2.VideoCapture(r"C:\Users\krish\flame_timelapse.mp4") # Here I passed time lapse video of burning diya 
cap.set(3,2000)
cap.set(4,2000)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model.predict(source=frame , conf=0.5, show=True)

    if len(results[0].boxes) > 0:
        for box in results[0].boxes:
            _, y1, _, y2 = box.xyxy[0] # Get the height of the bounding box
            flame_height = y2 - y1
            print(f"Flame height: {flame_height} pixels")
            
            if flame_height < 145:  # Beep alarm if flame height is less than 125 pixels
                print("Flame height is too low!")
                winsound.Beep(1000, 500)  # Frequency 1000 Hz, Duration 500 ms
            else:
                print("Flame detected with sufficient height")
    else:
        print("No flame detected.")
    
    
    cv2.imshow('Live Flame Detection', frame) # Display the results on the screen
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release() # Release the webcam and close all windows
cv2.destroyAllWindows()