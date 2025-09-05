import cv2
print(cv2.__version__)


IMGS_PATH = "../imgs"

snowman = cv2.imread(f"{IMGS_PATH}/snowman.png", cv2.IMREAD_GRAYSCALE)
futbol = cv2.imread(f"{IMGS_PATH}/futbol.jpg")

print("snowman.shape:", snowman.shape)
print("futbol.shape:", futbol.shape)

cv2.imshow("snowman", snowman)
cv2.imshow("futbol", futbol)
cv2.waitKey(0)
cv2.destroyAllWindows()
