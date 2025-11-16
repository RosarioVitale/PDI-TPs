import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("..")
from utils import fourier as uf

IMAGE_DIR = "../imgs/"
IMAGE_FILE1 = "puente.jpg"
IMAGE_FILE2 = "ferrari-c.png"
imagen1 = cv2.imread(f"{IMAGE_DIR}{IMAGE_FILE1}", cv2.IMREAD_GRAYSCALE)
imagen2 = cv2.imread(f"{IMAGE_DIR}{IMAGE_FILE2}", cv2.IMREAD_GRAYSCALE)

mag_img1, ph_img1, sh_img1 = uf.get_dft(imagen1)
mag_img2, ph_img2, sh_img2 = uf.get_dft(imagen2)

mg1, ph1 = cv2.cartToPolar(sh_img1[:,:,0], sh_img1[:,:,1])
mg2, ph2 = cv2.cartToPolar(sh_img2[:,:,0], sh_img2[:,:,1])
X1,Y1= cv2.polarToCart(mg1, ph2)
X2,Y2= cv2.polarToCart(mg2, ph1)
sh_img1=cv2.merge([X1,Y1])
sh_img2=cv2.merge([X2,Y2])

img_back1 = uf.get_idft(sh_img1)
img_back2 = uf.get_idft(sh_img2)

fig1, ax1 = plt.subplots(2, 4, layout="constrained")
fig1.suptitle("Ej: 2 Openheim", fontsize=16)
ax1[0][0].imshow(imagen1, cmap="gray", vmax=255, vmin=0)
ax1[0][0].set_title("Imagen 1")
ax1[1][0].imshow(imagen2, cmap="gray", vmax=255, vmin=0)
ax1[1][0].set_title("Imagen 2")
ax1[0][1].imshow(mag_img1, cmap="gray", vmax=255, vmin=0)
ax1[0][1].set_title("Magnitud imagen 1")
ax1[1][1].imshow(mag_img2, cmap="gray", vmax=255, vmin=0)
ax1[1][1].set_title("Magnitud imagen 2")
ax1[0][2].imshow(ph_img1, cmap="gray", vmax=255, vmin=0)
ax1[0][2].set_title("Fase imagen 1")
ax1[1][2].imshow(ph_img2, cmap="gray", vmax=255, vmin=0)
ax1[1][2].set_title("Fase imagen 2")
ax1[0][3].imshow(img_back1, cmap="gray", vmax=255, vmin=0)
ax1[0][3].set_title("Reconstruccion mag 1 ph 2")
ax1[1][3].imshow(img_back2, cmap="gray", vmax=255, vmin=0)
ax1[1][3].set_title("Reconstruccion mag 2 ph 1")

plt.show()
