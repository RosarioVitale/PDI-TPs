import cv2
print(cv2.__version__)


IMGS_PATH = "../imgs"
cv2.imshow("Hello World", cv2.imread(f"{IMGS_PATH}/snowman.png"))
cv2.waitKey(0)
cv2.destroyAllWindows()
