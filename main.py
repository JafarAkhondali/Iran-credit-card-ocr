import cv2
import math
import numpy as np


img_name = './dataset/melim.jpg'

def sortCorners(corners, center):
    top = []
    bot = []
    for c in corners:
        if c[1] < center[1]:
            top.append(c)
        else:
            bot.append(c)

    tl, tr = top
    bl, br = bot

    if tl[0] > tr[0]:
        tl, tr = tr, tl
    if bl[0] > br[0]:
        bl, br = br, bl

    return np.array([tl, tr, br, bl], dtype=np.float32)


def computeIntersect(a, b):
    x1 = a[0]
    y1 = a[1]
    x2 = a[2]
    y2 = a[3]
    x3 = b[0]
    y3 = b[1]
    x4 = b[2]
    y4 = b[3]

    d = (x1 - x2) * (y3 - y4) - ((y1 - y2) * (x3 - x4))
    if d != 0:
        point_x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) // d
        point_y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) // d
        if point_x < 0 or point_y < 0:
            return None
        return point_x, point_y
    return None


def near_point(a, b, threshhold):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) < threshhold


def blur(img, size):
    # return cv2.bilateralFilter(img, 25, 25, 75)  # source, kernel size
    return cv2.GaussianBlur(img, (size, size), 0)  # source, kernel size


def on_change(self=None):
    ksize = cv2.getTrackbarPos('ksize', 'Image')  # returns trackbar position
    ksize = ksize | 1
    median = blur(img_gray, ksize)
    showimg(median)


def showimg(img):
    shape = img.shape
    height = shape[0]
    width = shape[1]

    imgS = cv2.resize(img, (width, height))
    cv2.imshow("Image", imgS)
    k = cv2.waitKey(0) & 0xFF


img = cv2.imread(img_name)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

showimg(img)
# low_k = 1  # slider start position
# high_k = 21  # maximal slider position
cv2.namedWindow('Image')
# cv2.createTrackbar('ksize', 'Image', low_k, high_k, on_change)
# on_change()
blured = blur(img_gray, 5)
# showimg(blured)
edged_img = cv2.Canny(blured, 100, 100, 3)
showimg(edged_img)

lines = cv2.HoughLinesP(edged_img, 1, math.pi / 180, 90, 5, 10)

img_visual_lines = np.copy(img)
for l_ in lines:
    l = l_[0]
    cv2.line(img_gray, (l[0], l[1]), (l[2], l[3]), (0, 255, 0), 3)
print( "Count of lines: {}".format(len(lines)))
showimg(img_gray)
corners = []

lines_count = len(lines)

for i in range(lines_count):
    for j in range(i + 1, lines_count):
        pt = computeIntersect(lines[i][0], lines[j][0])
        if pt:
            any_center_nearby = False
            for c in corners:
                if near_point(pt, c, 20):
                    any_center_nearby = True
                    break
            if any_center_nearby:
                continue
            corners.append(pt)

if len(corners) != 4:
    print('Wrong number of corners')
    exit(0)

visualized_corners = np.copy(img)
for c in corners:
    cv2.circle(visualized_corners, c, 5, (255, 255, 120))

showimg(visualized_corners)

corners_center = [0, 0]
for c in corners:
    corners_center[0] += c[0]
    corners_center[1] += c[1]

corners_center[0] /= len(corners)
corners_center[1] /= len(corners)
corners = sortCorners(corners, corners_center)

# Wrap perspective image
w = 250
h = 150
cropped_cart = np.zeros((h, w, 3), np.uint8)
mapped_corners = [
    (0, 0),
    (w, 0),
    (w, h),
    (0, h)
]
mapped_corners = np.array(mapped_corners, dtype=np.float32)

transmtx = cv2.getPerspectiveTransform(corners, mapped_corners)
cropped_cart = cv2.warpPerspective(img, transmtx, (w, h) )
showimg(cropped_cart)

# hsv = cv2.cvtColor(cropped_cart, cv2.COLOR_BGR2HSV)
#
# lower_red = np.array([0, 0, 0])
# upper_red = np.array([180, 255, 120])
# mask = cv2.inRange(hsv, lower_red, upper_red)
lower_red = np.array([0, 0, 0])
upper_red = np.array([150, 150, 150])
mask = cv2.inRange(cropped_cart, lower_red, upper_red)
showimg(mask)



# cv::Point2f center(0,0);
# for (int i = 0; i < corners.size(); i++)
#     center += corners[i];
#
# center *= (1. / corners.size());
# sortCorners(corners, center);


# canny_edges = cv2.Canny(img, 150, 50)
#
# showimg(canny_edges)
# cv2.destroyAllWindows()
