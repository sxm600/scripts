import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


sift: cv.SIFT = cv.SIFT_create()
MIN_MATCH_COUNT = 10


def get_sift_descriptors(img):
    kp, des = sift.detectAndCompute(img, None)
    return kp, des


def get_matches(des1, des2, thresh = 0.75):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    return [m for m, n in matches if m.distance < thresh * n.distance]


def get_projection(src, dst, kp_src, kp_dst):
    src = src.copy()
    dst = dst.copy()
    src_pts = np.float32(kp_src).reshape(-1,1,2)
    dst_pts = np.float32(kp_dst).reshape(-1,1,2)

    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

    matches_mask = mask.ravel().tolist()

    h, w = src.shape
    pts = np.float32(
        [[  0,     0  ],
         [  0,   h - 1],
         [w - 1, h - 1],
         [w - 1,   0  ]]
    ).reshape(-1, 1, 2)

    projections = cv.perspectiveTransform(pts, M)

    dst = cv.polylines(dst,[np.int32(projections)],True,(0, 255, 0),3, cv.LINE_AA)

    return dst, matches_mask


def draw_matches(img1, img2, kp1, kp2, matches, matches_mask):
    draw_params = {
        'matchColor': (0, 255, 0),
        'singlePointColor': None,
        'matchesMask': matches_mask,
        'flags': 2
    }

    return cv.drawMatches(img1, kp1, img2, kp2, matches, None, **draw_params)


def main():
    img_train = cv.imread('../sources/feature/photo_2_train.jpg', cv.IMREAD_GRAYSCALE)
    img_test = cv.imread('../sources/feature/photo_2_query.jpg', cv.IMREAD_GRAYSCALE)

    kp_train, des_train = get_sift_descriptors(img_train)
    kp_test, des_test = get_sift_descriptors(img_test)

    matches = get_matches(des_train, des_test)

    if len(matches) > MIN_MATCH_COUNT:
        img_test, matches_mask = get_projection(
            img_train,
            img_test,
            [kp_train[m.queryIdx].pt for m in matches],
            [kp_test[m.trainIdx].pt for m in matches]
        )
    else:
        print(f'Not enough matches found (min={MIN_MATCH_COUNT}, found={len(matches)})')
        matches_mask = None

    img_out = draw_matches(img_train, img_test, kp_train, kp_test, matches, matches_mask)

    plt.imsave('../assets/feature-matching-panda.png', img_out)


if __name__ == "__main__":
    main()