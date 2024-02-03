import cv2
import os
import time
from ultralytics import YOLO


model = YOLO('hands_yolov8_2.pt')  # load a custom model


# Open a video capture object (you can replace '0' with the video file path)
cap = cv2.VideoCapture(0)

# Initialize a frame counter
frame_counter = 0


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame")
        break

    result = model.predict(frame, show=True)
    
    print(result[0].boxes)
    exit()
    # Display the frame (optional)
    # cv2.imshow('Frame', result)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture object and close any open windows
cap.release()
cv2.destroyAllWindows()
