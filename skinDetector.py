import numpy as np
import cv2

if __name__ == "__main__":
    # get the reference to the webcam
    camera = cv2.VideoCapture(0)
    while True:
        ret, frame = camera.read() #BGR
        #show img
        cv2.imshow("frame", frame)
        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF
        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            break
