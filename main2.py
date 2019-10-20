import numpy as np
import imutils
import cv2

cap = cv2.VideoCapture(0)
while True:
    # Capture frame-by-frame
    ret, frame = cap.read() #BGR
    blur = cv2.blur(frame, (5,5))
    cv2.imshow("blured", blur)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    cv2.imshow("binary", thresh)


    #Morphology operations
    kernel = np.ones((3,3), np.uint8)

    #dilation = cv2.dilate(thresh, kernel)
    #erosion = cv2.erode(dilation, kernel)

    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("closing ", closing)

    # observe the keypress by the user
    keypress = cv2.waitKey(1) & 0xFF
     # if the user pressed "q", then stop looping
    if keypress == ord("q"):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()