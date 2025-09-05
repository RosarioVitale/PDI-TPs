import cv2

ref = []

def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(ref) == 0:
            ref.append((x, y))
        else:
            cv2.line(img, ref[0], (x, y), (0, 0, 255), 5)
            ref.clear()

cv2.namedWindow("Hello World")
cv2.setMouseCallback("Hello World", click)

IMGS_PATH = "../imgs"
img = cv2.imread(f"{IMGS_PATH}/snowman.png", cv2.IMREAD_GRAYSCALE)
img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

stop = False
while not stop:
    cv2.imshow("Hello World", img)
    if cv2.waitKey(1) == ord("q"):
        stop = True
