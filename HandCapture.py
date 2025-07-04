import cv2
import mediapipe as mp
import time
# pyright: reportAttributeAccessIssue=false


cap=cv2.VideoCapture(0)

mpHands=mp.solutions.hands
hands=mpHands.Hands()

while True:
    success, img=cap.read()

    if not success:
        print("failed to load video capture")
        break

    cv2.imshow("Image",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
