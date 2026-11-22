
import cv2
import mediapipe as mp
import time
import math
import pyautogui

pyautogui.PAUSE = 0


# MediaPipe Tasks setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

import os

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models",
    "hand_landmarker.task"
)

# Hand landmark connections
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Webcam not detected!")
    exit()

print("Hand tracking started!")
print("Press Q to exit.")

frame_count = 0
start_time = time.monotonic()

previous_y = None
SCROLL_THRESHOLD = 0.015
SCROLL_AMOUNT = 5

with HandLandmarker.create_from_options(options) as landmarker:

    while True:
        success, frame = cap.read()

        if not success:
            print("Camera frame not available!")
            break

        frame = cv2.flip(frame, 1)

        height, width, _ = frame.shape

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB
        )

        # Create MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp_ms = int(
            (time.monotonic() - start_time) * 1000
        )

        # Detect hands
        results = landmarker.detect_for_video(
            mp_image, timestamp_ms
        )

        hand_count = len(results.hand_landmarks)

        cv2.putText(
            frame,
            f"Hands Detected: {hand_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Draw landmarks
        for hand in results.hand_landmarks:

            points = []

            for landmark in hand:
                x = int(landmark.x * width)
                y = int(landmark.y * height)

                points.append((x, y))

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )
            # Pinch gesture detection
            thumb_tip = hand[4]
            index_tip = hand[8]

            distance = math.hypot(
                thumb_tip.x - index_tip.x,
                thumb_tip.y - index_tip.y
            )

            if distance < 0.05:
                gesture = "PINCH DETECTED"
                color = (0, 255, 0)

                current_y = (thumb_tip.y + index_tip.y) / 2

                if previous_y is not None:
                    movement = previous_y - current_y

                    if movement > SCROLL_THRESHOLD:
                        pyautogui.scroll(SCROLL_AMOUNT)
                        print("Scrolling UP")

                    elif movement < -SCROLL_THRESHOLD:
                        pyautogui.scroll(-SCROLL_AMOUNT)
                        print("Scrolling DOWN")

                previous_y = current_y

            else:
                gesture = "NO PINCH"
                color = (0, 0, 255)
                previous_y = None

            cv2.putText(
                frame,
                gesture,
                (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2
            )
            # Draw hand connections
            for start, end in CONNECTIONS:
                cv2.line(
                    frame,
                    points[start],
                    points[end],
                    (255, 0, 0),
                    2
                )

        cv2.imshow("Touchless Scroller - Hand Tracking", frame)

        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

print("Hand tracking stopped!")

