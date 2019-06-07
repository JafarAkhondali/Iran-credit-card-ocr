import cv2
import math
import numpy as np
from collections import defaultdict
import pytesseract


# img_name = './dataset/saderat3.jpg'
# img_name = './dataset/saderat2.jpg'
# img_name = './dataset/saderat1.jpg'
# img_name = './dataset/bgwm.jpg'

# img_name = './dataset/keshm.jpg'
# img_name = './dataset/melim.jpg'
# img_name = './dataset/melatm2.jpg'
# img_name = './dataset/meli_persm.jpg'
# img_name = './dataset/meli_rotm.jpg'


#TODO: All of these are melat, so we have to specify ROI for Mellat

# img_name = './dataset/melatm3.jpg' #TODO CC
# img_name = './dataset/melatm.jpg' #TODO CC ( Works)
# img_name = './dataset/melatm.jpg' #TODO CC
img_name = './dataset/mininoise2.jpg'


# img_name = './dataset/noise3.jpg'


class LOGO:
    BANKS = {
        'TEJARAT': {
            'logo': cv2.imread('logos/tejarat.png', 0),
            'name': 'TEJARAT',
            'persian_name': 'TEJARAT',
        },
        'MELLI': {
            'logo': cv2.imread('logos/melli.png', 0),
            'name': 'MELLI',
            'persian_name': 'MELLI',
        },
        'SADERAT': {
            'logo': cv2.imread('logos/saderat.png', 0),
            'name': 'SADERAT',
            'persian_name': 'SADERAT',
        },
        'KESHAVARZI': {
            'logo': cv2.imread('logos/keshavarzi.png', 0),
            'name': 'KESHAVARZI',
            'persian_name': 'KESHAVARZI',
        },
    }


def segment_by_angle_kmeans(lines, k=2, **kwargs):
    """Groups lines based on angle with k-means.

    Uses k-means on the coordinates of the angle on the unit circle
    to segment `k` angles inside `lines`.
    """

    # Define criteria = (type, max_iter, epsilon)
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))
    flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
    attempts = kwargs.get('attempts', 10)

    # returns angles in [0, pi] in radians
    angles = np.array([line[0][1] for line in lines])
    # multiply the angles by two and find coordinates of that angle
    pts = np.array([[np.cos(2 * angle), np.sin(2 * angle)]
                    for angle in angles], dtype=np.float32)

    # run kmeans on the coords
    labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]
    labels = labels.reshape(-1)  # transpose to row vec

    # segment lines based on their kmeans label
    segmented = defaultdict(list)
    for i, line in zip(range(len(lines)), lines):
        segmented[labels[i]].append(line)
    segmented = list(segmented.values())
    return segmented


def intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [[x0, y0]]


def segmented_intersections(lines):
    """Finds the intersections between groups of lines."""

    intersections = []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i + 1:]:
            for line1 in group:
                for line2 in next_group:
                    intersections.append(intersection(line1, line2))

    return intersections


def sortCorners(corners, center):
    left = []
    right = []
    for c in corners:
        if c[0] < center[0]:
            left.append(c)
        else:
            right.append(c)

    tl, bl = left
    tr, br = right

    if tl[1] > bl[1]:
        tl, bl = bl, tl
    if tr[1] > br[1]:
        tr, br = br, tr

    return np.array([tl, tr, br, bl], dtype=np.float32)


# def sortCorners(corners, center):
#     top = []
#     bot = []
#     for c in corners:
#         if c[1] < center[1]:
#             top.append(c)
#         else:
#             bot.append(c)
#
#     tl, tr = top
#     bl, br = bot
#
#     if tl[0] > tr[0]:
#         tl, tr = tr, tl
#     if bl[0] > br[0]:
#         bl, br = br, bl
#
#     return np.array([tl, tr, br, bl], dtype=np.float32)
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
    # return cv2.bilateralFilter(img, size, size, 75)  # source, kernel size
    return cv2.GaussianBlur(img, (size, size), 0)  # source, kernel size


def bblur(img, size):
    return cv2.bilateralFilter(img, size, size, 75)  # source, kernel size


# def on_change(self=None):
#     ksize = cv2.getTrackbarPos('ksize', 'Image')  # returns trackbar position
#     ksize = ksize | 1
#     median = blur(img_gray, ksize)
#     showimg(median)


def showimg(img):
    shape = img.shape
    height = shape[0]
    width = shape[1]

    imgS = cv2.resize(img, (width, height))
    cv2.imshow("Image", imgS)
    k = cv2.waitKey(0) & 0xFF


img = cv2.imread(img_name)
image_height, image_width, _ = img.shape
total_count_of_pixels = image_height * image_width
showimg(img)
# low_k = 1  # slider start position
# high_k = 21  # maximal slider position
cv2.namedWindow('Image')
# cv2.createTrackbar('ksize', 'Image', low_k, high_k, on_change)
# on_change()
blured_img = bblur(img, 3)

# showimg(blured_img)
ggray = cv2.cvtColor(blured_img, cv2.COLOR_BGR2GRAY)
th, dif = cv2.threshold(ggray, 190, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
showimg(dif)
count_ones = cv2.countNonZero(dif)

# This is so important!
# What matters is that background should be in different color than cart
# And OTSU threshhold method will handle that, but sometimes cart turns into black instead of white
# We have to invert colors in threshold if this happens

# IF pixels are mostly white
if total_count_of_pixels//2 < count_ones:
    print("Inverting image")
    dif = cv2.bitwise_not(dif)
    showimg(dif)

kernel = np.ones((2, 2), np.uint8)

#
dif = cv2.morphologyEx(dif, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=5)
showimg(dif)

dif = cv2.erode(dif, np.ones((3, 3), np.uint8), iterations=3)
showimg(dif)

dif = cv2.morphologyEx(dif, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=4)
showimg(dif)

dif = cv2.morphologyEx(dif, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=10)
showimg(dif)

dif = cv2.morphologyEx(dif, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=5)
showimg(dif)

# dif = cv2.morphologyEx(dif, cv2.MORPH_OPEN, np.ones((4, 4), np.uint8), iterations=5)
# showimg(dif)
#
# dif = cv2.morphologyEx(dif, cv2.MORPH_CLOSE, np.ones((4, 4), np.uint8), iterations=5)
# showimg(dif)
#
# dif = cv2.morphologyEx(dif, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=3)
# showimg(dif)
#
# dif = cv2.morphologyEx(dif, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=3)
# showimg(dif)

# dif = cv2.morphologyEx(dif, cv2.MORPH_OPEN, kernel, iterations=3)
# showimg(dif)
"""
# bg = blur(blured_img, 201)
bg = bblur(blured_img, 25)
# dif = cv2.absdiff(blured_img, bg)
dif = cv2.absdiff(blured_img, bg)
print(1)
showimg(dif)

dif = blur(dif, 7)
print(2)
showimg(dif)
dif = cv2.addWeighted(img, .5, dif, 2.5, 0)

print(3)
showimg(dif)
dif = blur(dif, 7)
print(4)
# showimg(dif)
# dif = cv2.addWeighted(img, .5, dif, 2.5, 0)
dif = cv2.cvtColor(dif, cv2.COLOR_BGR2GRAY)
# showimg(dif)
# dif = cv2.adaptiveThreshold(dif, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
showimg(dif)
# kernel = np.ones((30, 30), np.uint8)
# dif = cv2.morphologyEx(dif, cv2.MORPH_OPEN, kernel)
# showimg(dif)

# # dif = cv2.morphologyEx(dif, cv2.MORPH_CLOSE, np.ones((200, 200), np.uint8))
# dif = cv2.morphologyEx(dif, cv2.MORPH_GRADIENT, np.ones((80, 80)))
# showimg(dif)
# mask = cv2.dilate(dif, kernel, iterations=2)
# showimg(mask)
# mask = cv2.erode(mask, np.ones((40, 40), np.uint8), iterations=2)
# showimg(mask)
# img = cv2.bitwise_and(img, img, mask=mask)
# showimg(img)
# img_gray = cv2.cvtColor(dif, cv2.COLOR_BGR2GRAY)
# blured = blur(img_gray, 1)
# showimg(blured)
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
"""
# img_gray = cv2.cvtColor(dif, cv2.COLOR_BGR2GRAY)

# dif = cv2.morphologyEx(dif, cv2.MORPH_RECT, np.ones((7, 7), np.uint8))
# showimg(dif)
# dif = cv2.morphologyEx(dif, cv2.MORPH_CROSS, np.ones((3, 3), np.uint8), iterations=10)
# showimg(dif)
edged_img = cv2.Canny(dif, 100, 100, 3)
showimg(edged_img)
countedges = cv2.countNonZero(edged_img)
# print(countedges)

# Dilate borders
# edged_img = cv2.morphologyEx(edged_img, cv2.MORPH_OPEN, np.ones((3, 3)))
# showimg(edged_img)

# edged_img= cv2.morphologyEx(edged_img, cv2.MORPH_CLOSE, kernel)
# showimg(edged_img)

edged_img = cv2.dilate(edged_img, np.ones((2, 2), np.uint8), iterations=1)
showimg(edged_img)

# edged_img = cv2.morphologyEx(edged_img, cv2.MORPH_TOPHAT, kernel)
# showimg(edged_img)


# lines = cv2.HoughLinesP(edged_img, 1, math.pi / 180, 100, 15, 10)

# img_visual_lines = np.copy(img)
# for l_ in lines:
#     l = l_[0]
#     cv2.line(img_visual_lines, (l[0], l[1]), (l[2], l[3]), (0, 255, 0), 1)
# print("Count of lines: {}".format(len(lines)))
# showimg(img_visual_lines)
# corners = []
# lines_count = len(lines)
# for i in range(lines_count):
#     for j in range(i + 1, lines_count):
#         pt = computeIntersect(lines[i][0], lines[j][0])
#         if pt:
#             any_center_nearby = False
#             for c in corners:
#                 if near_point(pt, c, 20):
#                     any_center_nearby = True
#                     break
#             if any_center_nearby:
#                 continue
#             corners.append(pt)

hough_threshhold = 60
while True:
    print("Trying %s for hough threshhold" % hough_threshhold)
    lines = cv2.HoughLines(edged_img, 1, math.pi / 180, hough_threshhold, 15, 10)
    print("Count of lines found: %s" % len(lines))

    if hough_threshhold < 10:
        print("Sorry, not today :(")
        break
    if len(lines) < 4:
        hough_threshhold -= 5
        continue
    else:
        segmented = segment_by_angle_kmeans(lines)
        intersections = segmented_intersections(segmented)
        print(len(intersections))
        break


corners = []
near_thresh = image_height//3

while True:
    print("Finding corners with threshold %s" % near_thresh)
    for i in intersections:
        if i[0][0] < 0 or i[0][1] < 0:
            continue
        any_center_nearby = False
        for c in corners:

            if near_point(i[0], c, near_thresh):
                any_center_nearby = True
                break
        if any_center_nearby:
            continue
        corners.append(tuple(i[0]))

    print("Current len of corners is %s " % len(corners))
    if near_thresh <= 0:
        break
    elif len(corners) < 4:
        near_thresh -= 50
        corners.clear()
    else:
        break

print("Len of corners: {}".format(len(corners)))

if len(corners) < 4:
    print('Wrong number of corners')
    exit(0)
corners = corners[:4]  # Take top four possibles, because it's sorted
visualized_corners = np.copy(img)
for c in corners:
    cv2.circle(visualized_corners, c, 5, (230, 50, 180), thickness=5)
showimg(visualized_corners)

corners_center = [0, 0]
for c in corners:
    corners_center[0] += c[0]
    corners_center[1] += c[1]

corners_center[0] /= len(corners)
corners_center[1] /= len(corners)
corners = sortCorners(corners, corners_center)

# Wrap perspective image
cc_width = 250
cc_height = 150
cc_image = np.zeros((cc_height, cc_width, 3), np.uint8)
mapped_corners = [
    (0, 0),
    (cc_width, 0),
    (cc_width, cc_height),
    (0, cc_height)
]
mapped_corners = np.array(mapped_corners, dtype=np.float32)

transmtx = cv2.getPerspectiveTransform(corners, mapped_corners)
cc_image = cv2.warpPerspective(img, transmtx, (cc_width, cc_height))
showimg(cc_image)


def find_bank(cc_image):
    h_cc, w_cc, _ = cc_image.shape
    roi_cc_image = cc_image[: 3*h_cc // 5, w_cc // 3:]
    logos_prob = []
    print('Finding bank logo')
    for key, value in LOGO.BANKS.items():
        cc_image_gray = cv2.cvtColor(roi_cc_image, cv2.COLOR_BGR2GRAY)
        template_w, template_h = value['logo'].shape[::-1]
        res = cv2.matchTemplate(cc_image_gray, value['logo'], cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = max_loc
        print(max_val)
        top_left = (top_left[0] + w_cc//3, top_left[1])
        bottom_right = (top_left[0] + template_w, top_left[1] + template_h)

        logos_prob.append({
            'value': max_val,
            'top_left': top_left,
            'bottom_right': bottom_right,
            'bank_key': key
        })

    max_val = logos_prob[0]['value']
    found_logo = logos_prob[0]
    # TODO: Set threshhold for bank logo
    for p in logos_prob[1:]:
        if p['value'] > max_val:
            max_val = p['value']
            found_logo = p

    bank = LOGO.BANKS[found_logo['bank_key']]
    bank_name = bank['name']
    cccopy = np.copy(cc_image)
    cv2.rectangle(cccopy, found_logo['top_left'], found_logo['bottom_right'], 255, 2)

    showimg(cccopy)
    return bank_name


bank_name = find_bank(cc_image)
print(bank_name)

# We are only interested on part of cart that contains number,
# on image with height of 150, the numbers should be on 80 - 125 range of H
numbers_part = cc_image[80:125, 10:-10]
# 80 125

hsvNumber = cv2.cvtColor(numbers_part, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsvNumber, (0, 0, 0), (180, 255, 70))
# mask = cv2.cvtColor(numbers_part, cv2.COLOR_BGR2HSV)
#
# lower_red = np.array([0, 0, 0])
# upper_red = np.array([180, 255, 120])
# mask = cv2.inRange(hsv, lower_red, upper_red)
# lower_red = np.array([0, 0, 0])
# upper_red = np.array([100, 100, 100])
# mask = cv2.inRange(numbers_part, lower_red, upper_red)
showimg(mask)

# contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

print("Number of contours: {}".format(len(contours)))

v_numbers_part = np.copy(numbers_part)
for c in contours:
    # cv2.drawContours(v_numbers_part, [c], 0, (randrange(30, 250), randrange(30, 250), randrange(30, 250)), 1)
    cv2.drawContours(v_numbers_part, [c], 0, (255, 255, 255), 1)
showimg(v_numbers_part)

realcontours = []
contours_rect = []
x_array = []
y_array = []

for c in contours:
    carea = cv2.contourArea(c)
    if carea >= 3 * 6 or carea == 0.0: # Sometimes carea is zero, contiue using rects
        [x, y, w, h] = cv2.boundingRect(c)
        if h > 3 and w > 2:
            if w * h >= 3 * 6:
                x_array.append(x + (w // 2))
                y_array.append(y + (h // 2))
                contours_rect.append([x, y, w, h])
                realcontours.append(c)

width_median = int(np.median(x_array))
height_median = int(np.median(y_array))

print("Number of improved contours: {}".format(len(realcontours)))
print(width_median)
print(height_median)

max_height = height_median + 14
min_height = height_median - 14
if min_height < 0:
    min_height = 0

cropped_mask = mask[min_height:max_height, :]
numbers_part = numbers_part[min_height:max_height, :]

contours, _ = cv2.findContours(cropped_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
v2_numbers_part = np.copy(numbers_part)
for c in contours:
    cv2.drawContours(v2_numbers_part, [c], 0, (255, 0, 0), 1)
print("Improved Contours again count: {}".format(len(contours)))
showimg(v2_numbers_part)

showimg(numbers_part)
config = ("-l eng --oem 1 --psm 7")
# config = ("-l eng --oem 1 --psm 7")
text = pytesseract.image_to_string(numbers_part, config=config)
validate_cc = text.replace(' ', '')
print(text)
if validate_cc.isnumeric() and len(validate_cc) == 16:
    print("CC Looks valid")
    print("CC Number: " + text)
else:
    print("CC Doesn't look valid :(")
    print("Let's select only numbers")
    digits = [s for s in validate_cc if s.isdigit()]
    digits = ''.join(digits)
    if digits.isnumeric() and len(digits) == 16:
        print("CC Looks valid")
        print("CC Number: " + digits)

# Remove small contours. With image size reduced to that state, we want at least 6x9 size for each number
