import cv2

IMGS_PATH = "../imgs"
img0 = cv2.imread(f"{IMGS_PATH}/snowman.png")
img1 = img0.copy()
img2 = img1[100:400, 100:400]
img2[:, 50:150] = 0

cv2.imshow("Hello World (img0)", img0)
cv2.imshow("Hello World (img1)", img1)
cv2.imshow("Hello World (img2)", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
