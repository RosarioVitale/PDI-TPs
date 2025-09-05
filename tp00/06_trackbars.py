import cv2

alpha_trackbar_max = 100

IMGS_PATH = "../imgs"
img1 = cv2.imread(f"{IMGS_PATH}/clown.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(f"{IMGS_PATH}/cameraman.tif", cv2.IMREAD_GRAYSCALE)

def on_trackbar(val):
    global img1, img2, dest
    alpha_trackbar = val / alpha_trackbar_max
    dest = cv2.addWeighted(img1, alpha_trackbar, img2, 1 - alpha_trackbar, 0)

cv2.namedWindow("Hello World")
cv2.createTrackbar("Alpha", "Hello World", 0, alpha_trackbar_max, on_trackbar)
dest = img2

while True:
    cv2.imshow("Hello World", dest)
    if cv2.waitKey(1) == ord("q"):
        break
