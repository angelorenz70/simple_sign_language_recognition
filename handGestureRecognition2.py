import mediapipe as mp
import cv2
import numpy as np
import uuid
import os
from landmark import Landmark
from ultralytics import YOLO

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hand_detection = YOLO('hand detection/hands_yolov8_2.pt')


# For static images:
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5)

cap = cv2.VideoCapture("videos/Manos.mov")

while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(25) & 0xFF == ord('r'):
      break

cap.release()
cv2.destroyAllWindows()