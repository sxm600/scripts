# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "opencv-python",
# ]
# ///

import sys

import cv2
import numpy as np


def get_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # bounds of red color on hsv model are around (0, 10) and (170, 180)
    mask = (
        cv2.inRange(hsv, (0, 200, 100), (10, 255, 255)) +
        cv2.inRange(hsv, (170, 200, 100), (180, 255, 255))
    )

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # remove black noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # remove white noise

    return mask


def draw_centroid(image, mask):
    M = cv2.moments(mask)

    if M['m00'] != 0.0:
        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])

        cv2.circle(image, (cX, cY), 5, (0, 255, 0), -1)


def draw_bounding_boxes(image, mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 255, 0), 2)
        cv2.putText(image,"Red Ball",(x - 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


def main():
    cap = cv2.VideoCapture('../sources/rgb_ball_720.mp4')

    if not cap.isOpened():
        sys.exit("Could not load the video.")

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        mask = get_mask(frame)
        draw_centroid(frame, mask)
        draw_bounding_boxes(frame, mask)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
