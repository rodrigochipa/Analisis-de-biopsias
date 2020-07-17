import cv2
import numpy as np

def find_blobs(contour_mask, minArea=25, minInertiaRatio=0.01, minThreshold=50, maxThreshold=200):
    # https://www.learnopencv.com/blob-detection-using-opencv-python-c/
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = False
    params.filterByCircularity = False
    params.filterByConvexity = False 
    params.filterByArea = True
    params.filterByInertia = True
    params.minThreshold = minThreshold
    params.maxThreshold = maxThreshold
    params.minArea = minArea
    params.minInertiaRatio = minInertiaRatio
    detector = cv2.SimpleBlobDetector_create(params)
    return detector.detect(contour_mask) # keypoints

def draw_keypoints(image, keypoints):
    return cv2.drawKeypoints(image, keypoints, np.array([]), (255,0,0), 
                             cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


def filter_image(image):
    result = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    result = cv2.medianBlur(result, 5)
    result = cv2.adaptiveThreshold(result, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 0)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, 
                                   kernel=np.ones((3, 3),np.uint8), iterations=5)
    return result
