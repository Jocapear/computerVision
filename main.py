import numpy as np
import imutils
import cv2
import face_recognition

bg = None
# --------------------------------------------------
# To find the running average over the background
# --------------------------------------------------


def run_avg(image, aWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, aWeight)

# ---------------------------------------------
# To segment the region of hand in the image
# ---------------------------------------------


def segment(image, threshold=15):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    (cnts, _) = cv2.findContours(thresholded.copy(),
                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)

# ---------------------------------------------
# To detect skin colors in the image
# ---------------------------------------------


def detect_skin(image):
    HSV_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    YCbCr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
    RGB_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    lower_HSV_values = np.array([0, 40, 0], dtype="uint8")
    upper_HSV_values = np.array([25, 105, 255], dtype="uint8")

    lower_YCbCr_values = np.array((0, 138, 67), dtype="uint8")
    upper_YCbCr_values = np.array((255, 173, 133), dtype="uint8")

    lower_RGB_values = np.array((95, 40, 20), dtype="uint8")
    upper_RGB_values = np.array((255, 255, 255), dtype="uint8")

    # A binary mask is returned. White pixels (255) represent pixels that fall into the upper/lower.
    mask_YCbCr = cv2.inRange(
        YCbCr_image, lower_YCbCr_values, upper_YCbCr_values)
    mask_HSV = cv2.inRange(
        HSV_image, lower_HSV_values, upper_HSV_values)
    mask_RGB = cv2.inRange(
        RGB_image, lower_RGB_values, upper_RGB_values)

    # binary_mask_image = cv2.add(cv2.add(mask_HSV, mask_YCbCr), mask_RGB)

    binary_mask_image = cv2.add(mask_HSV, mask_YCbCr)
    # binary_mask_image = cv2.bitwise_and(
    #     binary_mask_image, binary_mask_image, mask=mask_RGB)
    cv2.imshow("original_binary_image", binary_mask_image)
    binary_mask_image = cv2.GaussianBlur(binary_mask_image, (11, 11), 0)
    _, binary_mask_image = cv2.threshold(
        binary_mask_image, 128, 255, cv2.THRESH_BINARY)
    cv2.imshow("binary_image", binary_mask_image)
    return binary_mask_image
    '''
    kernel = np.ones((3, 3), np.uint8)
    kernel2 = np.ones((7, 7), np.uint8)
    image_foreground = cv2.erode(
        binary_mask_image, kernel, iterations=3)  # remove noise
    # The background region is reduced a little because of the dilate operation
    dilated_binary_image = cv2.dilate(
        binary_mask_image, kernel2, iterations=3)
    _, image_background = cv2.threshold(
        dilated_binary_image, 1, 128, cv2.THRESH_BINARY)  # set all background regions to 128

    # add both foreground and backgroud, forming markers. The markers are "seeds" of the future image regions.
    image_marker = cv2.add(image_foreground, image_background)
    image_marker32 = np.int32(image_marker)  # convert to 32SC1 format

    cv2.watershed(image, image_marker32)
    m = cv2.convertScaleAbs(image_marker32)  # convert back to uint8

    # bitwise of the mask with the input image
    _, image_mask = cv2.threshold(
        m, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return image_mask
    return cv2.bitwise_and(image, image, mask=image_mask)
    '''

# -----------------
# MAIN FUNCTION
# -----------------
if __name__ == "__main__":
    # initialize weight for running average
    aWeight = 0.5

    # get the reference to the webcam
    camera = cv2.VideoCapture(0)
    #camera = cv2.VideoCapture('http://192.168.1.96:8080/video')

    # region of interest (ROI) coordinates
    top, right, bottom, left = 0, 0, 700, 700
    #top, right, bottom, left = 10, 350, 225, 590

    # initialize num of frames
    num_frames = 0

    # keep looping, until interrupted
    while(True):
        # get the current frame
        (grabbed, frame) = camera.read()

        # resize the frame
        frame = imutils.resize(frame, width=700)

        # flip the frame so that it is not the mirror view
        frame = cv2.flip(frame, 1)

        # get the height and width of the frame
        (height, width) = frame.shape[:2]

        rgb_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        for (ftop, fright, fbottom, fleft) in face_locations:
            frame = cv2.rectangle(
                frame,
                (2 * fleft, 2 * ftop),
                (2 * fright, 2 * fbottom),
                (0, 0, 0),
                -1
            )

        # clone the frame
        clone = frame.copy()

        # get the ROI
        roi = frame[top:bottom, right:left]

        # convert the roi to grayscale and blur it
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # to get the background, keep looking till a threshold is reached
        # so that our running average model gets calibrated
        if num_frames < 30:
            run_avg(gray, aWeight)
        else:
            # segment the hand region
            hand = segment(gray)

            # check whether hand region is segmented
            if hand is not None:
                # if yes, unpack the thresholded image and
                # segmented region
                (thresholded, segmented) = hand

                # Morphologycal operations
                kernel = np.ones((7, 7), np.uint8)
                thresholded = cv2.morphologyEx(
                    thresholded,cv2.MORPH_CLOSE, kernel)

                # Detect skin
                skin_mask = detect_skin(clone)
                thresholded = cv2.bitwise_and(thresholded, thresholded, mask=skin_mask)

                # Find contours
                contours, _ = cv2.findContours(
                    thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                contours.sort(key=cv2.contourArea)
                contours.reverse()
                contours = contours[:2]
                hull_list = []

                # Find the convex hull object for each contour
                for i in range(len(contours)):
                    # if defects is not None:
                    hull = cv2.convexHull(contours[i])
                    hull_list.append(hull)

                    moments = cv2.moments(contours[i])
                    cx = 0
                    cy = 0
                    if moments['m00'] != 0:
                        cx = int(moments['m10']/moments['m00'])  # cx = M10/M00
                        cy = int(moments['m01']/moments['m00'])  # cy = M01/M00
                    centr = (cx, cy)
                    cv2.circle(clone, centr, 5, [0, 0, 255], 2)
                    cv2.drawContours(
                        thresholded, [contours[i]], 0, (0, 255, 0), 2)
                    cv2.drawContours(thresholded, [hull], 0, (0, 0, 255), 2)

                    cnt = cv2.approxPolyDP(
                        contours[i], 0.01*cv2.arcLength(contours[i], True), True)
                    hull = cv2.convexHull(cnt, returnPoints=False)
                    defects = cv2.convexityDefects(cnt, hull)
                    if defects is not None:
                        for i in range(defects.shape[0]):
                            s, e, f, d = defects[i, 0]
                            start = tuple(cnt[s][0])
                            end = tuple(cnt[e][0])
                            far = tuple(cnt[f][0])
                            cv2.pointPolygonTest(cnt, centr, True)
                            cv2.line(clone, start, end, [0, 255, 0], 2)
                            cv2.circle(clone, far, 5, [0, 0, 255], -1)

                # Draw contours + hull results
                drawing = np.zeros(
                    (thresholded.shape[0], thresholded.shape[1], 3), dtype=np.uint8)
                for i in range(len(contours)):
                    color = (0, 255, 255)
                    cv2.drawContours(drawing, contours, i, color)
                    cv2.drawContours(drawing, hull_list, i, color)
                cv2.imshow("Drawing", drawing)

                # draw the segmented region and display the frame
                cv2.drawContours(
                    clone, [segmented + (right, top)], -1, (0, 0, 255))
                cv2.imshow("Thesholded", thresholded)

        # draw the segmented hand
        cv2.putText(clone, str(num_frames), (0, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # increment the number of frames
        num_frames += 1

        # display the frame with segmented hand
        cv2.imshow("Video Feed", clone)

        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF

        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            break

# free up memory
camera.release()
cv2.destroyAllWindows()
