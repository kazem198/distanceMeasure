import cv2
import numpy as np
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector

cap = cv2.VideoCapture(0)
sen = 10
# cap.set(3, 1280)
# cap.set(4, 720)
detector = FaceMeshDetector(maxFaces=1)

textList = ["Welcome to ", "Murtaza's Workshop.",
            "Here we will study", "Computer Vision,", "Robotics and AI.",
            "If you like this video", "Like, Share", "and Subscribe."]

while True:
    _, img = cap.read()
    imgText = np.zeros_like(img)

    img, faces = detector.findFaceMesh(img, draw=False)
    if faces:
        for face in faces:
            leftEye = face[145]
            rightEye = face[374]

            cv2.line(img, leftEye, rightEye, (0, 200, 0), 5)
            cv2.circle(img, rightEye, 5, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, leftEye, 5, (255, 0, 255), cv2.FILLED)

            w, _ = detector.findDistance(leftEye, rightEye)
            W = 6.3
        #     d = 50
        #     f = (w*d)/W
        #     print(f)
            f = 840
            d = (W*f)/w
        #     print(d)

            cvzone.putTextRect(
                img, f'depth is {int(d)}cm', (face[10][0]-100, face[10][1]-50), scale=2)

            for i, text in enumerate(textList):
                singleHeight = 20 + int((int(d/sen)*sen)/4)
                scale = 0.4 + (int(d/sen)*sen)/75
                cv2.putText(imgText, text, (50, 50 + (i * singleHeight)),
                            cv2.FONT_ITALIC, scale, (255, 255, 255), 2)

        imgStatck = cvzone.stackImages([img, imgText], 2, 1)
        cv2.imshow("img", imgStatck)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        break
