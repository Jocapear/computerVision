#!/usr/bin/env python3

import numpy as np
import cv2
from PIL import Image, ImageEnhance


def enhance_image(image):
    pil_im = Image.fromarray(image)
    contrast = ImageEnhance.Contrast(pil_im)
    contrast = contrast.enhance(1)
    return np.array(contrast)


def detect_skin(image):
    HSV_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    YCbCr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)

    lower_HSV_values = np.array([0, 40, 0], dtype="uint8")
    upper_HSV_values = np.array([25, 255, 255], dtype="uint8")

    lower_YCbCr_values = np.array((0, 138, 67), dtype="uint8")
    upper_YCbCr_values = np.array((255, 173, 133), dtype="uint8")

    # A binary mask is returned. White pixels (255) represent pixels that fall into the upper/lower.
    mask_YCbCr = cv2.inRange(
        YCbCr_image, lower_YCbCr_values, upper_YCbCr_values)
    mask_HSV = cv2.inRange(
        HSV_image, lower_HSV_values, upper_HSV_values)

    binary_mask_image = cv2.add(mask_HSV, mask_YCbCr)
    image_foreground = cv2.erode(
        binary_mask_image, None, iterations=3)  # remove noise
    # The background region is reduced a little because of the dilate operation
    dilated_binary_image = cv2.dilate(
        binary_mask_image, None, iterations=3)
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
    return cv2.bitwise_and(image, image, mask=image_mask)


if __name__ == "__main__":
    # get the reference to the webcam
    camera = cv2.VideoCapture(0)
    while True:
        ret, frame = camera.read()  # BGR
        frame = cv2.flip(frame, 1)
        cv2.imshow("initial", frame)
        frame = enhance_image(frame)
        cv2.imshow("enhanced", frame)
        result = detect_skin(frame)
        # show img
        cv2.imshow("result", result)
        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF
        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            break
