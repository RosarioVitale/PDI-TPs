import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from utils import utils as ut

IMAGE_DIR = "../imgs/"
IMAGE_FILE = "patron.tif"
imagen = cv2.imread(f"{IMAGE_DIR}{IMAGE_FILE}")

imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
R = imagen[:,:,0]
G = imagen[:,:,1]
B = imagen[:,:,2]
seg_r = ut.get_segment(imagen_rgb, 1, R.shape[0]//2, R.shape[1], R.shape[0]//2)

imagen_hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
H = imagen_hsv[:,:,0]
S = imagen_hsv[:,:,1]
V = imagen_hsv[:,:,2]
seg_h = ut.get_segment(imagen_hsv, 1, R.shape[0]//2, R.shape[1], R.shape[0]//2)

fig, ax = plt.subplots(2, 5)
ax[0, 0].imshow(imagen_rgb)
ax[0, 0].set_title("RGB")
ax[0, 1].imshow(R, vmin=0, vmax=255, cmap="gray")
ax[0, 1].set_title("R")
ax[0, 2].imshow(G, vmin=0, vmax=255, cmap="gray")
ax[0, 2].set_title("G")
ax[0, 3].imshow(B, vmin=0, vmax=255, cmap="gray")
ax[0, 3].set_title("B")
ax[0, 4].plot(seg_r[:, 0], "r", label="R")
ax[0, 4].plot(seg_r[:, 1], "g", label="G")
ax[0, 4].plot(seg_r[:, 2], "b", label="B")
ax[0, 4].legend()
ax[1, 0].imshow(imagen_hsv)
ax[1, 0].set_title("HSV")
ax[1, 1].imshow(H, vmin=0, vmax=255, cmap="gray")
ax[1, 1].set_title("H")
ax[1, 2].imshow(S, vmin=0, vmax=255, cmap="gray")
ax[1, 2].set_title("S")
ax[1, 3].imshow(V, vmin=0, vmax=255, cmap="gray")
ax[1, 3].set_title("V")
ax[1, 4].plot(seg_h[:, 0], "r", label="H")
ax[1, 4].plot(seg_h[:, 1], "g", label="S")
ax[1, 4].plot(seg_h[:, 2], "b", label="V")
ax[1, 4].legend()
plt.show()
