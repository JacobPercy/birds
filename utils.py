import cv2
import numpy as np
#import rembg

CANNY_THRESH1 = 80
CANNY_THRESH2 = 200
BLUR_SIZE = 5
DILATION_KERNEL_SIZE = 3
DILATION_ITERATIONS = 1

#def remove_bg(img):
#    return rembg.remove(img)

def sketch(frame):
    #frame = remove_bg(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (BLUR_SIZE, BLUR_SIZE), 0)
    edges = cv2.Canny(blurred, CANNY_THRESH1, CANNY_THRESH2)
    kernel = np.ones((DILATION_KERNEL_SIZE, DILATION_KERNEL_SIZE), np.uint8)
    sketch = cv2.dilate(edges, kernel, iterations=DILATION_ITERATIONS)
    return sketch