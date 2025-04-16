import os

import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_tray_contour(img_gray):
    ret, thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return max(contours, key=cv2.contourArea)


def get_coins_circles(img_gray):
    params = {
        "method": cv2.HOUGH_GRADIENT,
        "dp": 1,  # accumulator
        "minDist": 20,  # minimum distance between centers
        "param1": 20,  # threshold for Canny Edge Detector
        "param2": 56,  # method specific parameter
        "minRadius": 10,
        "maxRadius": 50
    }

    circles, *_ = cv2.HoughCircles(img_gray, **params)

    return np.uint16(np.round(circles))


def draw_tray_contour(img_original, contour):
    area = cv2.contourArea(contour)
    cv2.drawContours(img_original, [contour], 0, (0, 255, 0), 3)
    cv2.putText(img_original, f"Area: {area}", (50, 50), 0, 1, (255, 255, 255), 3)


def draw_coin_circles(img_original, circles):
    for x, y, radius in circles:
        cv2.circle(img_original, (x, y), radius, (0, 255, 0), 2)  # draw circumference
        cv2.circle(img_original, (x, y), 2, (0, 0, 255), 3)  # draw center


def draw_different_circles_count(img_original, circles, tray_contour):
    big_tray = 0
    small_tray = 0

    big_not_tray = 0
    small_not_tray = 0

    big_radius_threshold = 31

    for x, y, radius in circles:
        if cv2.pointPolygonTest(tray_contour, (x, y), False) > -1:
            if radius > big_radius_threshold:
                big_tray += 1
            else:
                small_tray += 1
        else:
             if radius > big_radius_threshold:
                 big_not_tray += 1
             else:
                 small_not_tray += 1

    cv2.putText(img_original, f"Big tray: {big_tray}", (50, 100), 0, 1, (255, 255, 255), 3)
    cv2.putText(img_original, f"Small tray: {small_tray}", (50, 150), 0, 1, (255, 255, 255), 3)
    cv2.putText(img_original, f"Big not tray: {big_not_tray}", (50, 200), 0, 1, (255, 255, 255), 3)
    cv2.putText(img_original, f"Small not tray: {small_not_tray}", (50, 250), 0, 1, (255, 255, 255), 3)


def main():
    fig, axs = plt.subplots(2, 4, figsize=(38.40, 21.60))

    for idx, img_path in enumerate(os.listdir("../sources/tray")):
        img_original = cv2.imread(f"../sources/tray/{img_path}")
        img_blur = cv2.medianBlur(img_original, 3)  # 2n + 1; 1 <= n <= 10
        img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)

        tray_contour = get_tray_contour(img_gray)
        circles = get_coins_circles(img_gray)

        draw_different_circles_count(img_original, circles, tray_contour)
        draw_tray_contour(img_original, tray_contour)
        draw_coin_circles(img_original, circles)

        ax = axs[idx // 4, idx % 4]
        ax.imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
        ax.set_title(img_path, fontsize=16)

    fig.tight_layout()
    fig.savefig("coin-detection-plot.png")


if __name__ == "__main__":
    main()