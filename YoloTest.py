from ultralytics import YOLO
import cv2
import time
import math
import numpy as n
from gameLogic import TouchGame

# Load model/pose model
model = YOLO("yolov8n-pose.pt")

# Open webcam
cap = cv2.VideoCapture(0)

seen_id = set()

# Create window
window_name = "YOLO Detection"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Fullscreen
cv2.setWindowProperty(
    window_name,
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

game = TouchGame(duration=5)

paused = False
fullscreen = True

print("Controls:")
print("P - Pause/Play")
print("Q - Quit")
print("F - Fullscreen")
print("S - Save Screenshot")

def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

TOUCH_THRESHOLD = 50

while True:

    if not paused:

        success, frame = cap.read()

        if not success:
            print("Failed to capture")
            break

        # Run YOLO
        results = model(frame)

        # Draw detections
        annotated_frame = results[0].plot()

        game.update_target()

        players = []

        # TARGET (FIXED DISPLAY)
        cv2.putText(
            annotated_frame,
            f"TOUCH YOUR {game.current_target}",
            (frame.shape[1]//2 - 200, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        # SCORE DISPLAY (FIXED)
       
        cv2.putText(
            annotated_frame,
            f"P1: {game.scores['Player 1']}  |  P2: {game.scores['Player 2']}",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        # KEYPOINTS from COCO
        if results[0].keypoints is not None:

            keypoints = results[0].keypoints.xy.cpu().numpy()

            for person in keypoints:

                nose = person[0]
                left_shoulder = person[5]
                right_shoulder = person[6]
                left_wrist = person[9]
                right_wrist = person[10]
                left_knee = person[13]
                right_knee = person[14]
                left_ankle = person[15]
                right_ankle = person[16]

                wrists = [left_wrist, right_wrist]

                detections = []

                for wrist in wrists:

                   # FIXED LABELS (CRITICAL)
                    
                    if distance(nose, wrist) < TOUCH_THRESHOLD:
                        detections.append("HEAD")

                    if (distance(left_shoulder, wrist) < TOUCH_THRESHOLD or
                        distance(right_shoulder, wrist) < TOUCH_THRESHOLD):
                        detections.append("SHOULDERS")

                    if (distance(left_knee, wrist) < TOUCH_THRESHOLD or
                        distance(right_knee, wrist) < TOUCH_THRESHOLD):
                        detections.append("KNEES")

                    if (distance(left_ankle, wrist) < TOUCH_THRESHOLD or
                        distance(right_ankle, wrist) < TOUCH_THRESHOLD):
                        detections.append("TOES")

                center_x = nose[0]

                detections = list(set(detections))

                players.append({
                    "x": center_x,
                    "detections": detections
                })

            players = sorted(players, key=lambda p: p["x"])

            for i, player in enumerate(players[:2]):

                player_name = f"Player {i+1}"
                y = 120

                # SCORING
                scored = game.check_winner(player_name, player["detections"])

                if i == 0:
                    x = 20
                    color = (0, 255, 0)
                else:
                    x = frame.shape[1] - 220
                    color = (255, 0, 0)

                cv2.putText(
                    annotated_frame,
                    player_name,
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

                y += 30

                for text in player["detections"]:
                    cv2.putText(
                        annotated_frame,
                        text,
                        (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2
                    )
                    y += 25

                # FIXED INDENTATION (was broken before)
                if scored:
                    cv2.putText(
                        annotated_frame,
                        "NICE!",
                        (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )
                    y += 30

    cv2.imshow(window_name, annotated_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    elif key == ord("f"):
        fullscreen = not fullscreen

        if fullscreen:
            cv2.setWindowProperty(
                window_name,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN
            )
        else:
            cv2.setWindowProperty(
                window_name,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_NORMAL
            )

    elif key == ord("p"):
        paused = not paused

cap.release()
cv2.destroyAllWindows()