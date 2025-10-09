import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from utils import color as co
from utils import filters as fi

IMAGE_DIR = "../imgs/"
IMAGE_FILE = "camino.tif"
imagen = cv2.imread(f"{IMAGE_DIR}{IMAGE_FILE}")

###############################################################################
# Filtrado en RGB
###############################################################################
imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
R = imagen_rgb[:,:,0]
G = imagen_rgb[:,:,1]
B = imagen_rgb[:,:,2]
R_f = cv2.filter2D(R, -1, fi.generateKernel(fi.KBOXHI1, (3,3)))
G_f = cv2.filter2D(G, -1, fi.generateKernel(fi.KBOXHI1, (3,3)))
B_f = cv2.filter2D(B, -1, fi.generateKernel(fi.KBOXHI1, (3,3)))
filtrada_rgb = cv2.merge([R_f, G_f, B_f])
fig0, ax0 = plt.subplots(2, 4)
ax0[0,0].imshow(imagen_rgb)
ax0[0,0].set_title("Original")
ax0[0,1].imshow(R)
ax0[0,1].set_title("Canal R")
ax0[0,2].imshow(G)
ax0[0,2].set_title("Canal G")
ax0[0,3].imshow(B)
ax0[0,3].set_title("Canal B")
ax0[1,0].imshow(filtrada_rgb)
ax0[1,0].set_title("Resultado")
ax0[1,1].imshow(R_f)
ax0[1,1].set_title("Canal R filtrado")
ax0[1,2].imshow(G_f)
ax0[1,2].set_title("Canal G filtrado")
ax0[1,3].imshow(B_f)
ax0[1,3].set_title("Canal B filtrado")

###############################################################################
# Equalizacion de histograma en HSV
###############################################################################
imagen_hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
H = imagen_hsv[:,:,0]
S = imagen_hsv[:,:,1]
V = imagen_hsv[:,:,2]
H_f = cv2.filter2D(H, -1, fi.generateKernel(fi.KBOXHI1, (3,3)))
S_f = cv2.filter2D(S, -1, fi.generateKernel(fi.KBOXHI1, (3,3)))
V_f = cv2.filter2D(V, -1, fi.generateKernel(fi.KBOXHI1, (3,3)))
filtrada_hsv = cv2.merge([H_f, S_f, V_f])
fig1, ax1 = plt.subplots(2, 4)
ax1[0,0].imshow(imagen_rgb)
ax1[0,0].set_title("Original")
ax1[0,1].imshow(H)
ax1[0,1].set_title("Canal H")
ax1[0,2].imshow(S)
ax1[0,2].set_title("Canal S")
ax1[0,3].imshow(V)
ax1[0,3].set_title("Canal V")
ax1[1,0].imshow(cv2.cvtColor(filtrada_hsv, cv2.COLOR_HSV2BGR)[:,:,::-1])
ax1[1,0].set_title("Resultado")
ax1[1,1].imshow(H_f)
ax1[1,1].set_title("Canal H filtrado")
ax1[1,2].imshow(S_f)
ax1[1,2].set_title("Canal S filtrado")
ax1[1,3].imshow(V_f)
ax1[1,3].set_title("Canal V filtrado")

fig2, ax2 = plt.subplots(1, 3)
ax2[0].imshow(imagen_rgb)
ax2[0].set_title("Original")
ax2[1].imshow(filtrada_rgb)
ax2[1].set_title("Filtrada RGB")
ax2[2].imshow(cv2.cvtColor(filtrada_hsv, cv2.COLOR_HSV2BGR)[:,:,::-1])
ax2[2].set_title("Filtrada HSV")

plt.show()
