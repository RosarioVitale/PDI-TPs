import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from utils import color as co
from utils import histogram as hi

IMAGE_DIR = "../imgs/"
IMAGE_FILE = "chairs_oscura.jpg"
imagen = cv2.imread(f"{IMAGE_DIR}{IMAGE_FILE}")
imagen_deseada = cv2.cvtColor(cv2.imread(f"{IMAGE_DIR}chairs.jpg"), 
                              cv2.COLOR_BGR2RGB)

###############################################################################
# Equalizacion de histograma en RGB
###############################################################################
imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
R = imagen_rgb[:,:,0]
G = imagen_rgb[:,:,1]
B = imagen_rgb[:,:,2]
hist_r, hist_g, hist_b = hi.get_histogram(imagen_rgb, mode="BGR")
equalizada_rgb = hi.equalize_hist(imagen_rgb, mode="BGR")
R_eq = equalizada_rgb[:,:,0]
G_eq = equalizada_rgb[:,:,1]
B_eq = equalizada_rgb[:,:,2]
hist_r_eq, hist_g_eq, hist_b_eq = hi.get_histogram(equalizada_rgb, mode="BGR")
fig0, ax0 = plt.subplots(3, 5)
ax0[0,0].imshow(imagen_rgb)
ax0[0,0].set_title("Original")
ax0[1,0].imshow(equalizada_rgb)
ax0[1,0].set_title("Resultado")
ax0[2,0].imshow(imagen_deseada)
ax0[2,0].set_title("Deseada")
ax0[0,1].imshow(R)
ax0[0,1].set_title("Canal R")
ax0[1,1].imshow(G)
ax0[1,1].set_title("Canal G")
ax0[2,1].imshow(B)
ax0[2,1].set_title("Canal B")
ax0[0,2].stem(hist_r)
ax0[0,2].set_title("Histograma R")
ax0[1,2].stem(hist_g)
ax0[1,2].set_title("Histograma G")
ax0[2,2].stem(hist_b)
ax0[2,2].set_title("Histograma B")
ax0[0,3].imshow(R_eq)
ax0[0,3].set_title("Canal R ecualizado")
ax0[1,3].imshow(G_eq)
ax0[1,3].set_title("Canal G ecualizado")
ax0[2,3].imshow(B_eq)
ax0[2,3].set_title("Canal B ecualizado")
ax0[0,4].stem(hist_r_eq)
ax0[0,4].set_title("Histograma R ecualizado")
ax0[1,4].stem(hist_g_eq)
ax0[1,4].set_title("Histograma G ecualizado")
ax0[2,4].stem(hist_b_eq)
ax0[2,4].set_title("Histograma B ecualizado")


###############################################################################
# Equalizacion de histograma en HSV
###############################################################################
imagen_hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
H = imagen_hsv[:,:,0]
S = imagen_hsv[:,:,1]
V = imagen_hsv[:,:,2]
hist_h, hist_s, hist_v = hi.get_histogram(imagen_hsv, mode="HSV")
equalizada_hsv = hi.equalize_hist(imagen_hsv, mode="HSV", channels=[2])
H_eq = equalizada_hsv[:,:,0]
S_eq = equalizada_hsv[:,:,1]
V_eq = equalizada_hsv[:,:,2]
hist_h_eq, hist_s_eq, hist_v_eq = hi.get_histogram(equalizada_hsv, mode="HSV")
fig1, ax1 = plt.subplots(3, 5)
ax1[0,0].imshow(imagen_rgb)
ax1[0,0].set_title("Original")
ax1[1,0].imshow(cv2.cvtColor(equalizada_hsv, cv2.COLOR_HSV2BGR)[:,:,::-1])
ax1[1,0].set_title("Resultado")
ax1[2,0].imshow(imagen_deseada)
ax1[2,0].set_title("Deseada")
ax1[0,1].imshow(H)
ax1[0,1].set_title("Canal H")
ax1[1,1].imshow(S)
ax1[1,1].set_title("Canal S")
ax1[2,1].imshow(V)
ax1[2,1].set_title("Canal V")
ax1[0,2].stem(hist_h)
ax1[0,2].set_title("Histograma H")
ax1[1,2].stem(hist_s)
ax1[1,2].set_title("Histograma S")
ax1[2,2].stem(hist_v)
ax1[2,2].set_title("Histograma V")
ax1[0,3].imshow(H_eq)
ax1[0,3].set_title("Canal H ecualizado")
ax1[1,3].imshow(S_eq)
ax1[1,3].set_title("Canal S ecualizado")
ax1[2,3].imshow(V_eq)
ax1[2,3].set_title("Canal V ecualizado")
ax1[0,4].stem(hist_h_eq)
ax1[0,4].set_title("Histograma H ecualizado")
ax1[1,4].stem(hist_s_eq)
ax1[1,4].set_title("Histograma S ecualizado")
ax1[2,4].stem(hist_v_eq)
ax1[2,4].set_title("Histograma V ecualizado")

###############################################################################
# Equalizacion de histograma en HSI
###############################################################################
imagen_hsi = co.RGB2HSI(imagen_rgb)
H = imagen_hsi[:,:,0]
S = imagen_hsi[:,:,1]
I = imagen_hsi[:,:,2]
hist_h, hist_s, hist_i = hi.get_histogram(imagen_hsi, mode="HSV")
equalizada_hsi = hi.equalize_hist(imagen_hsi, mode="HSV", channels=[2])
H_eq = equalizada_hsi[:,:,0]
S_eq = equalizada_hsi[:,:,1]
I_eq = equalizada_hsi[:,:,2]
hist_h_eq, hist_s_eq, hist_i_eq = hi.get_histogram(equalizada_hsi, mode="HSV")

fig2, ax2 = plt.subplots(3, 5)
ax2[0,0].imshow(imagen_rgb)
ax2[0,0].set_title("Original")
ax2[1,0].imshow(co.HSI2RGB(equalizada_hsi))
ax2[1,0].set_title("Resultado")
ax2[2,0].imshow(imagen_deseada)
ax2[2,0].set_title("Deseada")
ax2[0,1].imshow(H)
ax2[0,1].set_title("Canal H")
ax2[1,1].imshow(S)
ax2[1,1].set_title("Canal S")
ax2[2,1].imshow(I)
ax2[2,1].set_title("Canal I")
ax2[0,2].stem(hist_h)
ax2[0,2].set_title("Histograma H")
ax2[1,2].stem(hist_s)
ax2[1,2].set_title("Histograma S")
ax2[2,2].stem(hist_i)
ax2[2,2].set_title("Histograma I")
ax2[0,3].imshow(H_eq)
ax2[0,3].set_title("Canal H ecualizado")
ax2[1,3].imshow(S_eq)
ax2[1,3].set_title("Canal S ecualizado")
ax2[2,3].imshow(I_eq)
ax2[2,3].set_title("Canal I ecualizado")
ax2[0,4].stem(hist_h_eq)
ax2[0,4].set_title("Histograma H ecualizado")
ax2[1,4].stem(hist_s_eq)
ax2[1,4].set_title("Histograma S ecualizado")
ax2[2,4].stem(hist_i_eq)
ax2[2,4].set_title("Histograma I ecualizado")

plt.show()
