import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from cvui import cvui
from utils import color as co

IMAGE_DIR = "../imgs/"
IMAGE_FILE = "rosas.jpg"
#IMAGE_FILE = "pattern.tif"
imagen = cv2.imread(f"{IMAGE_DIR}{IMAGE_FILE}")
img_cm = co.complementary_color(imagen)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
ax[0].set_title("Original")
ax[1].imshow(cv2.cvtColor(img_cm, cv2.COLOR_BGR2RGB))
ax[1].set_title("Complementario")
plt.show()
