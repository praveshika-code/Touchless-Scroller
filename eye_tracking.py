
import cv2
import mediapipe as mp
import os
import time
import pyautogui

pyautogui.PAUSE = 0

last_blink_time = 0
BLINK_COOLDOWN = 1.0

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models",
    "face_landmarker.task"
)

options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1,
    output_face_blendshapes=True
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Webcam not detected!")
    exit()

print("Eye tracking started!")
print("Press Q to exit.")

start_time = time.monotonic()

with FaceLandmarker.create_from_options(options) as landmarker:

    while True:
        success, frame = cap.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp_ms = int(
            (time.monotonic() - start_time) * 1000
        )

        results = landmarker.detect_for_video(
            mp_image, timestamp_ms
        )

        
        if results.face_landmarks:
            cv2.putText(
                frame,
                "FACE DETECTED",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            if results.face_blendshapes:
                blendshapes = results.face_blendshapes[0]

                scores = {
                    item.category_name: item.score
                    for item in blendshapes
                }

                left_blink = scores.get("eyeBlinkLeft", 0)
                right_blink = scores.get("eyeBlinkRight", 0)



                if left_blink > 0.5 and right_blink > 0.5:
                    status = "BLINK DETECTED"
                    color = (0, 255, 0)

                    current_time = time.monotonic()

                    if current_time - last_blink_time > BLINK_COOLDOWN:
                        pyautogui.scroll(-5)
                        last_blink_time = current_time
                        print("Scrolling DOWN")

                else:
                    status = "EYES OPEN"
                    color = (255, 255, 0)
                

                cv2.putText(
                    frame,
                    status,
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color,
                    2
                )


        else:
            cv2.putText(
                frame,
                "NO FACE DETECTED",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        cv2.imshow("Touchless Scroller - Eye Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

print("Eye tracking stopped!")
