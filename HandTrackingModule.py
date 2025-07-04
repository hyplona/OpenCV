import cv2
import mediapipe as mp
import time
import mediapipe.python.solutions.drawing_utils
import mediapipe.python.solutions.hands

class handDetector():
    def __init__(self,mode=False, maxHands=2, detectionCon=0.5,trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon=0.5
        self.trackCon=0.5

        self.mpHands = mp.solutions.hands
        self.hands = mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)

#formal CONFIG add this everytime
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

#for fps counter
pTime=0
cTime=0

# Custom drawing style
#hand_landmark_style = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=2)
#hand_connection_style = mpDraw.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to load video capture")
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB) #convert to rgb

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                print(id,lm)
                h, w, c=img.shape
                cx, cy = int(lm.x*w),int(lm.y*h)
                print(id, cx, cy)
                if id==4:
                    cv2.circle(img, (cx, cy), 15, (255,0,255), cv2.FILLED) #marks the point with the id in if
            mpDraw.draw_landmarks(
                img,
                handLms,
                mpHands.HAND_CONNECTIONS #for lines
            )

    #for fps counter
    cTime=time.time()
    fps =1/(cTime-pTime)
    pTime=cTime

    cv2.putText(img, str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3) #decimal to int to string, position, font, size, color, thickness

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

def main():
    #for fps counter
    pTime=0
    cTime=0
    cap = cv2.VideoCapture(1)

    # Custom drawing style
    #hand_landmark_style = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=2)
    #hand_connection_style = mpDraw.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1)

    while True:
        success, img = cap.read()

        cTime=time.time()
        fps =1/(cTime-pTime)
        pTime=cTime

        cv2.putText(img, str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3) #decimal to int to string, position, font, size, color, thickness

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



if __name__ =="__main__":
    main()
