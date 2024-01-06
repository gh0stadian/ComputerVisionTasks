import cv2
import numpy as np

# For Harris corner detector we opted to use a block size of 2 (blockSize=2), which means the size of the pixels
# to look at corner detection and a ksize of 3 (ksize=3) to use a 3x3 kernel for the Sobel operator.
# We also used a k=0.04 to detect as most corners as possible
def get_haris_kp_img(rgb_image):
    image = rgb_image.copy()
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_image = np.float32(gray_image)

    dst = cv2.cornerHarris(gray_image, blockSize=2, ksize=3, k=0.04)

    # dilate to mark the corners
    dst = cv2.dilate(dst, None)
    image[dst > 0.01 * dst.max()] = [0, 255, 0]

    kp = np.uint8(np.argwhere(dst > 0.01 * dst.max()))
    return [cv2.KeyPoint(coord[1], coord[0], 1) for coord in kp], image

def get_sift_kp_img(rgb_image, nfeatures=None):
    image = rgb_image.copy()
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    sift = cv2.SIFT_create()
    kp = sift.detect(gray_image, None)
    kp_img = cv2.drawKeypoints(image, kp, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return kp, kp_img

def get_fast_kp_img(rgb_image, threshold=None):
    image = rgb_image.copy()
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    fast = cv2.FastFeatureDetector_create(threshold=threshold)
    kp = fast.detect(gray_image, None)
    kp_image = cv2.drawKeypoints(image, kp, None, color=(0, 255, 0))
    return kp, kp_image

def get_orb_kp_img(rgb_image, nfeatures=None):
    image = rgb_image.copy()
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    orb = cv2.ORB_create(nfeatures=nfeatures)
    kp = orb.detect(gray_image, None)
    kp_image = cv2.drawKeypoints(image, kp, None, color=(0, 255, 0), flags=0)
    return kp, kp_image

def compute_sift_des(gray_img, kp, nfeatures=None):
    sift = cv2.SIFT_create(nfeatures=nfeatures)
    _, des = sift.compute(gray_img, kp)
    return des

def compute_orb_des(gray_img, kp, nfeatures=None):
    orb = cv2.ORB_create(nfeatures=nfeatures)
    _, des = orb.compute(gray_img, kp)
    return des

# For brute force matcher we opted to use it with KnnMatch with 2 nearest neighbors (k=2)
def knn_matcher(des_lookup, des_patch):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_patch, des_lookup, k=2)
    good_matches = []
    for m1, m2 in matches:
        if m1.distance < 0.6 * m2.distance:
            good_matches.append([m1])

    return good_matches

# For flann matcher we opted to use Hierarchical clustering index as the algorithm (algorithm=1) and 5 trees (trees=5)
# and 50 checks (checks=50) to balance the speed and accuracy of the matcher.
def flann_matcher(des_lookup, des_patch):
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = flann.knnMatch(des_patch, des_lookup, k=2)
    good_matches = []
    for m1, m2 in matches:
        if m1.distance < 0.6 * m2.distance:
            good_matches.append([m1])

    return good_matches