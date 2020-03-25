import cv2
import math
import numpy as np
from collections import defaultdict
from skimage.feature import hog
from sklearn.externals import joblib
import sys

knn = joblib.load('knn_model.pkl')


class LOGO:
    BANKS = {
        'TEJARAT': {
            'logo': cv2.imread('logos/tejarat.png', 0),
            'name': 'TEJARAT',
            'persian_name': 'TEJARAT',
            'numPos': {
                'from': (104, 288,),
                'to': (912, 440,)
            }
        },
        'MELLI': {
            'logo': cv2.imread('logos/melli.png', 0),
            'name': 'MELLI',
            'persian_name': 'MELLI',
            'numPos': {
                'from': (90, 343,),
                'to': (886, 443,)
            }
        },
        'SADERAT': {
            'logo': cv2.imread('logos/saderat.png', 0),
            'name': 'SADERAT',
            'persian_name': 'SADERAT',
            'numPos': {
                'from': (95, 306,),
                'to': (874, 408,)
            }
        },
        'KESHAVARZI': {
            'logo': cv2.imread('logos/keshavarzi.png', 0),
            'name': 'KESHAVARZI',
            'persian_name': 'KESHAVARZI',
            'numPos': {
                'from': (52, 385),
                'to': (918, 465,)
            }
        },
    }


def predict_knn(df):
    predict = knn.predict(df.reshape(1, -1))[0]
    predict_proba = knn.predict_proba(df.reshape(1, -1))
    return predict, predict_proba[0][predict]


def inputdata(inputimage):
    (H, hogImage) = hog(
        inputimage,
        orientations=8,
        pixels_per_cell=(10, 10),
        cells_per_block=(7, 5),
        transform_sqrt=True,
        block_norm="L1",
        visualize=True
    )
    return predict_knn(H)


def segment_by_angle_kmeans(lines, k=2, **kwargs):
    """Groups lines based on angle with k-means.
    Uses k-means on the coordinates of the angle on the unit circle
    to segment `k` angles inside `lines`.
    """

    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))
    flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
    attempts = kwargs.get('attempts', 10)

    angles = np.array([line[0][1] for line in lines])

    pts = np.array([[np.cos(2 * angle), np.sin(2 * angle)]
                    for angle in angles], dtype=np.float32)

    labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]
    labels = labels.reshape(-1)

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
    return cv2.GaussianBlur(img, (size, size), 0)


def bblur(img, size):
    return cv2.bilateralFilter(img, 3 * size, 3 * size, 175)


def try_ocr(img):
    img_copy = np.copy(img)
    img_copy = cv2.resize(img_copy, (50, 70))
    classify = inputdata(img_copy)
    print('=========')
    print("Classified as [%s] with our method" % (classify[0]))
    return str(classify[0])


def trim(im, left=True, right=True, top=True, down=True, open=True):
    img = np.copy(im)
    img = cv2.morphologyEx(im, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    l = t = 0
    b, r = img.shape
    thresh = 1
    img_h, img_w = img.shape
    if left:
        for i in range(0, img_w, 1):
            if np.count_nonzero(img[:, i]) > thresh:
                l = i - 1
                break
    if right:
        for i in range(img_w - 1, -1, -1):
            x = np.count_nonzero(img[:, i])
            if x > thresh:
                r = i + 2
                break
    if top:
        for i in range(0, img_h, 1):
            if np.count_nonzero(img[i, :]) > thresh:
                t = i - 1
                break
    if down:
        for i in range(img_h - 1, 0, -1):
            if np.count_nonzero(img[i, :]) > thresh:
                b = i + 1
                break
    if t < 0:
        t = 0
    if l < 0:
        l = 0
    if r >= img.shape[1]:
        r = img.shape[1] - 1
    if b >= img.shape[0]:
        b = img.shape[0] - 1
    return img[t:b, l:r]


def split_horizontal(img, start, end):
    im = np.copy(img)
    _, part_width = im.shape
    im = cv2.morphologyEx(im, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    search_range = 3
    search_range_start = end + search_range
    if search_range_start >= part_width:
        search_range_start = part_width - 1
    min_index = search_range_start
    min_val = np.count_nonzero(img[:, search_range_start])
    search_range_to = end - search_range + 1
    for i in range(search_range_start, search_range_to, -1):
        lmin = np.count_nonzero(img[:, i])
        if lmin < min_val:
            min_val = lmin
            min_index = i
    return im[:, start:min_index], min_index + 1


def showimg(img):
    cv2.imshow("Image", img)
    cv2.waitKey(0) & 0xFF


def find_bank(cc_image):
    h_cc, w_cc, _ = cc_image.shape
    roi_cc_image = cc_image[: 3 * h_cc // 5, w_cc // 3:]
    logos_prob = []
    print('Finding bank logo')
    for key, value in LOGO.BANKS.items():
        cc_image_gray = cv2.cvtColor(roi_cc_image, cv2.COLOR_BGR2GRAY)
        template_w, template_h = value['logo'].shape[::-1]
        res = cv2.matchTemplate(cc_image_gray, value['logo'], cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = max_loc
        print(max_val)
        top_left = (top_left[0] + w_cc // 3, top_left[1])
        bottom_right = (top_left[0] + template_w, top_left[1] + template_h)
        logos_prob.append({
            'value': max_val,
            'top_left': top_left,
            'bottom_right': bottom_right,
            'bank_key': key
        })
    max_val = logos_prob[0]['value']
    found_logo = logos_prob[0]

    for p in logos_prob[1:]:
        if p['value'] > max_val:
            max_val = p['value']
            found_logo = p
    if max_val < 0.50:
        print(" I think i have failed to find log of bank :(")
    else:
        bank = LOGO.BANKS[found_logo['bank_key']]
        bank_name = bank['name']
        cccopy = np.copy(cc_image)
        cv2.rectangle(cccopy, found_logo['top_left'], found_logo['bottom_right'], 255, 2)
        showimg(cccopy)
        return bank_name


def run(img_name):
    img = cv2.imread(img_name)
    image_height, image_width, _ = img.shape
    total_count_of_pixels = image_height * image_width
    showimg(img)
    cv2.namedWindow('Image')
    blured_img = bblur(img, 3)
    base_vue = 120
    ggray = cv2.cvtColor(blured_img, cv2.COLOR_BGR2GRAY)
    th, dif = cv2.threshold(ggray, 150, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    showimg(dif)
    count_ones = cv2.countNonZero(dif)
    if total_count_of_pixels // 2 < count_ones:
        print("Inverting image")
        base_vue = 150
        dif = cv2.bitwise_not(dif)
        showimg(dif)
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

    edged_img = cv2.Canny(dif, 100, 100, 3)
    showimg(edged_img)
    edged_img = cv2.dilate(edged_img, np.ones((2, 2), np.uint8), iterations=1)
    showimg(edged_img)
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
    near_thresh = image_height // 3
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
    corners = corners[:4]
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
    cc_width = 1000
    cc_height = 600
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

    bank_name = find_bank(cc_image)
    print(bank_name)
    x1, y1, x2, y2 = (20, 280, -20, 540)
    if bank_name and LOGO.BANKS[bank_name]['numPos']:
        x1, y1 = LOGO.BANKS[bank_name]['numPos']['from']
        x2, y2 = LOGO.BANKS[bank_name]['numPos']['to']
    numbers_part = cc_image[y1 - 10:y2 + 10, x1:x2]
    hsvNumber = cv2.cvtColor(numbers_part, cv2.COLOR_BGR2HSV)
    while True:
        print("Filtering black")
        mask = cv2.inRange(hsvNumber, (0, 0, 0), (200, 255, base_vue))
        if np.count_nonzero(mask) < ((hsvNumber.shape[0] * hsvNumber.shape[1] * 4) / 10):
            break
        base_vue -= 20
        print("Too DARK!")
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    print("Number of contours: {}".format(len(contours)))
    if len(contours) > 1:
        v_numbers_part = np.copy(numbers_part)
        for c in contours:
            cv2.drawContours(v_numbers_part, [c], 0, (255, 255, 255), 1)
        showimg(v_numbers_part)
    showimg(numbers_part)
    realcontours = []
    contours_rect = []
    x_array = []
    y_array = []
    if len(contours) > 1:
        for c in contours:
            carea = cv2.contourArea(c)
            if carea >= 12 * 24 or carea == 0.0:
                [x, y, w, h] = cv2.boundingRect(c)
                if h > 12 and w > 8:
                    if w * h >= 12 * 24:
                        x_array.append(x + (w // 2))
                        y_array.append(y + (h // 2))
                        contours_rect.append([x, y, w, h])
                        realcontours.append(c)
        height_median = int(np.median(y_array))
        print("Number of improved contours: {}".format(len(realcontours)))
        max_height = height_median + 30
        min_height = height_median - 30
        if min_height < 0:
            min_height = 0
        if max_height >= numbers_part.shape[0]:
            max_height = numbers_part.shape[0] - 1
        print(max_height)
        print(min_height)

        numbers_part = numbers_part[min_height:max_height, :]
    hsv_number_part_final = cv2.cvtColor(np.copy(numbers_part), cv2.COLOR_BGR2HSV)
    bw_number_part = cv2.inRange(hsv_number_part_final, (0, 0, 0), (200, 255, base_vue))
    showimg(bw_number_part)
    bw_number_part = trim(bw_number_part, open=False)
    showimg(bw_number_part)
    parts_height, parts_width = bw_number_part.shape
    part_width = (parts_width - 1) // 4
    p1, next = split_horizontal(bw_number_part, 0, part_width)
    p2, next = split_horizontal(bw_number_part, next, next + part_width)
    p3, next = split_horizontal(bw_number_part, next, next + part_width)
    p4, next = split_horizontal(bw_number_part, next, parts_width + 2)
    parts = [p1, p2, p3, p4]
    cc = ""
    for i, part in enumerate(parts):
        print("Showing part {}".format(i))
        part = trim(part, open=False)
        showimg(part)
        digit_group_height, digit_group_width = part.shape
        digit_width = (digit_group_width - 1) // 4
        d1, next = split_horizontal(part, 0, digit_width)
        showimg(d1)
        d1 = trim(d1)
        c = try_ocr(d1)
        cc += c
        showimg(d1)
        d2, next = split_horizontal(part, next, next + digit_width)
        showimg(d2)
        d2 = trim(d2)
        c = try_ocr(d2)
        cc += c
        showimg(d2)
        d3, next = split_horizontal(part, next, next + digit_width)
        showimg(d3)
        d3 = trim(d3)
        c = try_ocr(d3)
        cc += c
        showimg(d3)
        d4, next = split_horizontal(part, next, digit_group_width + 5)
        showimg(d4)
        d4 = trim(d4)
        c = try_ocr(d4)
        cc += c + " "
        showimg(d4)
    print("Final cc: %s" % cc)
    showimg(img)

    return cc, bank_name


if len(sys.argv) != 2:
    print("Usage: python {} path_of_input_image".format(sys.argv[0]))
    exit()

img_name = sys.argv[1]
cc, bank_name = run(img_name)
