import cv2
from fingertrack import HandDetector
import time
import sys

print("Python path:", sys.executable)

# Camera
cap = cv2.VideoCapture(0)

pTime = 0
detector = HandDetector()

while True:
    success, img = cap.read()
    if not success:
        print("Camera not detected")
        break

    # Detect hand
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    if len(lmList) != 0:
        fingers = detector.fingersUp()
        totalFingers = fingers.count(1)

        # Display finger count
        cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
        cv2.putText(
            img,
            str(totalFingers),
            (45, 375),
            cv2.FONT_HERSHEY_PLAIN,
            10,
            (255, 0, 0),
            5
        )

    # FPS calculation
    cTime = time.time()
    fps = 1 / (cTime - pTime) if cTime != pTime else 0
    pTime = cTime

    cv2.putText(
        img,
        f'FPS: {int(fps)}',
        (400, 70),
        cv2.FONT_HERSHEY_PLAIN,
        3,
        (255, 0, 0),
        3
    )

    cv2.imshow("Finger Count", img)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
