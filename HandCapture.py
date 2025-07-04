import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

# Custom drawing style
#hand_landmark_style = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=2)
#hand_connection_style = mpDraw.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to load video capture")
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(
                img,
                handLms
            )

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
