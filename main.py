import cv2


def showimg(img):
    height, width, _ = img.shape
    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    imgS = cv2.resize(img, (width // 3, height // 3))
    cv2.imshow("output", imgS)
    cv2.waitKey(0)


img = cv2.imread('./dataset/meli.jpg')
showimg(img)


cv2.destroyAllWindows()
