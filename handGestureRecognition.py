import mediapipe as mp
import cv2
import numpy as np
from landmark import Landmark
# from ultralytics import YOLO
import torch
import pickle

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
# hand_detection = YOLO('hand detection/hands_yolov8_2.pt')


cap = cv2.VideoCapture(2)
cv2.namedWindow('Hand Tracking', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Hand Tracking', 800, 600)

def points_sequences(hand_keypoints_landmark,xmin, ymin, width, height):
    xy = [xmin, ymin]
    wh = [width, height]
    hand_keypoints_landmark = torch.tensor(hand_keypoints_landmark)
    xy = torch.tensor(xy)
    wh = torch.tensor(wh)

    relative_hand_keypoints = hand_keypoints_landmark - xy
    relative_hand_keypoints = relative_hand_keypoints / wh

    return relative_hand_keypoints

label = 3
dataset_filename = f'three_label{label}'
dataset_length = 500
datasets = []


with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    landmark_location:Landmark = Landmark()

    while cap.isOpened():
        ret, frame = cap.read()
        
        # BGR 2 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Flip on horizontal
        image = cv2.flip(image, 1)
        image = cv2.resize(image, (800, 600))


        # print(image.shape)
        # Set flag
        image.flags.writeable = False
        
        # Detections
        results = hands.process(image)
        
        # Set flag to true
        image.flags.writeable = True
        
        # RGB 2 BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Rendering results
        if results.multi_hand_landmarks:
            # image = result_hand_detection[0].plot()
            for num, hand in enumerate(results.multi_hand_landmarks):
                landmark_location.set_landmarks(hand.landmark)
                xmin,ymin,xmax,ymax = landmark_location.get_bbox_coordinates(image.shape)
                xminn,yminn,xmaxn,ymaxn = landmark_location.get_bbox_normalized(image.shape)
                centerxn,centeryn,widthn,heightn = landmark_location.get_bbox_coordinates_xywh(image.shape)
                hand_keypoints = landmark_location.list_of_landmarks_xy

                cv2.putText(image, dataset_filename, (int(xmin), int(ymin)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


                #generate datasets start ------------
                datapoints = points_sequences(hand_keypoints, xminn, yminn, widthn, heightn)
                datasets.append((datapoints, label))
                print(f"length {len(datasets)}  datapoints size {datapoints.size()}")
                if len(datasets) == dataset_length:
                    with open(f"datasets/{dataset_filename}.pkl", "wb") as write:
                        pickle.dump(datasets, write)
                        print(f"{dataset_filename} datasets successfully saved!")
                        exit()
                #generate datasets end -----------


                cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
                # print(landmark_location.bbox, " normalize: ", landmark_location.get_bbox_normalized(image.shape))

                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(250, 0, 250), thickness=2, circle_radius=2),
                                         )
                
                # Displaying coordinates on the image
                # cv2.putText(image, f'X center : {xminn}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                # cv2.putText(image, f'Y center: {yminn}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                # cv2.circle(image, (int(xminn * image.shape[0]), int(yminn * image.shape[1])), 10, (255,12,115), -1)


            
        
        cv2.imshow('Hand Tracking', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()