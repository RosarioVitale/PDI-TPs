import cv2
import matplotlib.pyplot as plt

IMGS_PATH = "../imgs"
img = cv2.imread(f"{IMGS_PATH}/snowman.png", cv2.IMREAD_GRAYSCALE)
img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

cv2.line(img, (0, 0), (img.shape[1], img.shape[0]), (0, 0, 255), 5)
cv2.rectangle(img, (0, 0), (img.shape[1] // 2, img.shape[0] // 2), (0, 255, 0), 5)
cv2.circle(img, (img.shape[1] // 2, img.shape[0] // 2), 100, (255, 0, 0), 5)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

cv2.imshow("Hello World", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
