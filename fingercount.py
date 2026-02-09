import cv2
from fingertrack import HandDetector
import time
import os
import sys
print("Python path:", sys.executable)


pTime = 0
cap = cv2.VideoCapture(0)

folderPath = r'images'   # ✅ change ONLY if path differs
myList = os.listdir(folderPath)
print("Images found:", myList)
overlayList = []

for imPath in myList:
    image = cv2.imread(f"{folderPath}/{imPath}")
    overlayList.append(image)

detector = HandDetector()

while True:
    success, img = cap.read()
    if not success:
        break

    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    if len(lmList) != 0:
        fingers = detector.fingersUp()
        totalFingers = fingers.count(1)

        # Prevent index error
        if totalFingers < len(overlayList):
            h, w, c = overlayList[totalFingers].shape
            img[0:h, 0:w] = overlayList[totalFingers]

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

    cTime = time.time()
    fps = 1 / (cTime - pTime)
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

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
