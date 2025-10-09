import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from utils import color as co

IMAGE_DIR = "../imgs/"
IMAGE_FILE = "patron.tif"
imagen = cv2.imread(f"{IMAGE_DIR}{IMAGE_FILE}")

imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
R = imagen[:,:,0]
G = imagen[:,:,1]
B = imagen[:,:,2]

imagen_hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
H = imagen_hsv[:,:,0]
S = imagen_hsv[:,:,1]
V = imagen_hsv[:,:,2]

fig, ax = plt.subplots(2, 4)
ax[0, 0].imshow(imagen_rgb)
ax[0, 0].set_title("RGB")
ax[0, 1].imshow(R)
ax[0, 1].set_title("R")
ax[0, 2].imshow(G)
ax[0, 2].set_title("G")
ax[0, 3].imshow(B)
ax[0, 3].set_title("B")
ax[1, 0].imshow(imagen_hsv)
ax[1, 0].set_title("HSV")
ax[1, 1].imshow(H)
ax[1, 1].set_title("H")
ax[1, 2].imshow(S)
ax[1, 2].set_title("S")
ax[1, 3].imshow(V)
ax[1, 3].set_title("V")
plt.show()
