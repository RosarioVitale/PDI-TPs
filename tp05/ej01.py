import cv2
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append("..")
from utils import fourier as uf
#import imutils

IMG_SZ, SQR_L, CIR_R, REC_A, REC_B = 256, 100, 50, 50, 150
IMG_SZ2, DEG = 512, 20

###############################################################################
#                                Draw figures                                 #
###############################################################################
############
#  Item a  #
############
im_lin_h = np.zeros((IMG_SZ, IMG_SZ))
im_lin_h[IMG_SZ//2,:]=255
im_lin_v = np.zeros((IMG_SZ, IMG_SZ))
im_lin_v[:,IMG_SZ//2]=255
im_sqr = np.zeros((IMG_SZ, IMG_SZ))
im_sqr[(IMG_SZ-SQR_L)//2:(IMG_SZ+SQR_L)//2,
       (IMG_SZ-SQR_L)//2:(IMG_SZ+SQR_L)//2] = 255
im_rec = np.zeros((IMG_SZ, IMG_SZ))
im_rec[(IMG_SZ-REC_A)//2:(IMG_SZ+REC_A)//2,
       (IMG_SZ-REC_B)//2:(IMG_SZ+REC_B)//2] = 255
im_cir = np.zeros((IMG_SZ, IMG_SZ))
cv2.circle(im_cir, (IMG_SZ//2,IMG_SZ//2), CIR_R, [255], cv2.FILLED)

############
#  Item b  #
############
im_lin_b = np.zeros((IMG_SZ2, IMG_SZ2))
im_lin_b[:,IMG_SZ2//2]=255
#im_lin_b = imutils.rotate(im_lin_b, DEG)
center = (IMG_SZ2//2,IMG_SZ2//2)
M = cv2.getRotationMatrix2D(center, DEG, 1.0)
im_lin_b = cv2.warpAffine(im_lin_b, M, (IMG_SZ2, IMG_SZ2)) # img rotada
#im_lin_b2 = imutils.rotate(im_lin_b, 45)
M = cv2.getRotationMatrix2D(center, 45, 1.0)
im_lin_b2 = cv2.warpAffine(im_lin_b, M, (IMG_SZ2, IMG_SZ2))

im_lin_b = im_lin_b[(IMG_SZ2-IMG_SZ)//2:(IMG_SZ2+IMG_SZ)//2,
                    (IMG_SZ2-IMG_SZ)//2:(IMG_SZ2+IMG_SZ)//2]
#im_lin_b2 = im_lin_b2[(IMG_SZ2-IMG_SZ)//2:(IMG_SZ2+IMG_SZ)//2,
#                      (IMG_SZ2-IMG_SZ)//2:(IMG_SZ2+IMG_SZ)//2]
IMAGE_DIR = "../imgs/"
IMAGE_FILE1 = "clown.jpg"
IMAGE_FILE2 = "cameraman.tif"
imagen1 = cv2.imread(f"{IMAGE_DIR}{IMAGE_FILE1}", cv2.IMREAD_GRAYSCALE)
imagen2 = cv2.imread(f"{IMAGE_DIR}{IMAGE_FILE2}", cv2.IMREAD_GRAYSCALE)

###############################################################################
#                                   Fourier                                   #
###############################################################################

fim_lin_h,fim_lin_h_p,_ = uf.get_dft(im_lin_h)
fim_lin_v,fim_lin_v_p,_ = uf.get_dft(im_lin_v)
fim_sqr,fim_sqr_p,_ = uf.get_dft(im_sqr)
fim_rec,fim_rec_p,_ = uf.get_dft(im_rec)
fim_cir,fim_cir_p,_ = uf.get_dft(im_cir)

fim_lin_b,fim_lin_b_p,_ = uf.get_dft(im_lin_b)
#fim_lin_b2,_ = uf.get_dft(im_lin_b2)
fimagen1,fimagen1_p,_ = uf.get_dft(imagen1)
fimagen2,fimagen2_p,_ = uf.get_dft(imagen2)

###############################################################################
#                                 Matplotlib                                  #
###############################################################################
##############
#  Figure 1  #
##############
fig1, ax1 = plt.subplots(3, 5, layout="constrained")
fig1.suptitle("Ej: 1 Imagenes binarias 1", fontsize=16)
ax1[0][0].imshow(im_lin_h, cmap="gray", vmax=255, vmin=0)
ax1[0][0].set_title("Linea horizontal")
ax1[0][1].imshow(im_lin_v, cmap="gray", vmax=255, vmin=0)
ax1[0][1].set_title("Linea vertical")
ax1[0][2].imshow(im_sqr, cmap="gray", vmax=255, vmin=0)
ax1[0][2].set_title("Cuadrado")
ax1[0][3].imshow(im_rec, cmap="gray", vmax=255, vmin=0)
ax1[0][3].set_title("Rectangulo")
ax1[0][4].imshow(im_cir, cmap="gray", vmax=255, vmin=0)
ax1[0][4].set_title("Circulo")

ax1[1][0].imshow(fim_lin_h, cmap="gray", vmax=255, vmin=0)
ax1[1][0].set_title("Linea horizontal: DFT")
ax1[1][1].imshow(fim_lin_v, cmap="gray", vmax=255, vmin=0)
ax1[1][1].set_title("Linea vertical: DFT")
ax1[1][2].imshow(fim_sqr, cmap="gray", vmax=255, vmin=0)
ax1[1][2].set_title("Cuadrado: DFT")
ax1[1][3].imshow(fim_rec, cmap="gray", vmax=255, vmin=0)
ax1[1][3].set_title("Rectangulo: DFT")
ax1[1][4].imshow(fim_cir, cmap="gray", vmax=255, vmin=0)
ax1[1][4].set_title("Circulo: DFT")

ax1[2][0].imshow(fim_lin_h_p, cmap="gray", vmax=255, vmin=0)
ax1[2][0].set_title("Linea horizontal: Fase")
ax1[2][1].imshow(fim_lin_v_p, cmap="gray", vmax=255, vmin=0)
ax1[2][1].set_title("Linea vertical: Fase")
ax1[2][2].imshow(fim_sqr_p, cmap="gray", vmax=255, vmin=0)
ax1[2][2].set_title("Cuadrado: Fase")
ax1[2][3].imshow(fim_rec_p, cmap="gray", vmax=255, vmin=0)
ax1[2][3].set_title("Rectangulo: Fase")
ax1[2][4].imshow(fim_cir_p, cmap="gray", vmax=255, vmin=0)
ax1[2][4].set_title("Circulo: Fase")

##############
#  Figure 2  #
##############
fig2, ax2 = plt.subplots(3, 3, layout="constrained")
fig2.suptitle("Ej: 1 Imagenes binarias 2", fontsize=16)
ax2[0][0].imshow(im_lin_b, cmap="gray", vmax=255, vmin=0)
ax2[0][0].set_title("Linea rotada")
ax2[1][0].imshow(fim_lin_b, cmap="gray", vmax=255, vmin=0)
ax2[1][0].set_title("Linea rotada: DFT")
ax2[2][0].imshow(fim_lin_b_p, cmap="gray", vmax=255, vmin=0)
ax2[2][0].set_title("Linea rotada: Fase")
ax2[0][1].imshow(imagen1, cmap="gray", vmax=255, vmin=0)
ax2[0][1].set_title("Imagen 1")
ax2[1][1].imshow(fimagen1, cmap="gray", vmax=255, vmin=0)
ax2[1][1].set_title("Imagen 1: DFT")
ax2[2][1].imshow(fimagen1_p, cmap="gray", vmax=255, vmin=0)
ax2[2][1].set_title("Imagen 1: Fase")
ax2[0][2].imshow(imagen2, cmap="gray", vmax=255, vmin=0)
ax2[0][2].set_title("Imagen 2")
ax2[1][2].imshow(fimagen2, cmap="gray", vmax=255, vmin=0)
ax2[1][2].set_title("Imagen 2: DFT")
ax2[2][2].imshow(fimagen2_p, cmap="gray", vmax=255, vmin=0)
ax2[2][2].set_title("Imagen 2: Fase")

plt.show()
