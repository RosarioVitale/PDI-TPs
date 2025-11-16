import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("..")
from utils import fourier as uf

IMAGE_DIR = "../imgs/"
IMAGE_FILE1 = "cameraman.tif"
imagen1 = cv2.imread(f"{IMAGE_DIR}{IMAGE_FILE1}", cv2.IMREAD_GRAYSCALE)
SIGMA = 10
ftype = uf.GAUSSIAN_FILTER

mag_img1, ph_img1, sh_img1 = uf.get_dft(imagen1)
params_id, is_lowpass = {"sigma":SIGMA}, True
filt1 = uf.gaussian_filter(imagen1.shape, params_id, is_lowpass)
img_filt1 = uf.apply_filter(imagen1, ftype, params_id, is_lowpass)

fig1, ax1 = plt.subplots(1, 3, layout="constrained")
fig1.suptitle("Ej: 3 Filtros", fontsize=16)
ax1[0].imshow(imagen1, cmap="gray", vmax=255, vmin=0)
ax1[0].set_title("Imagen 1")
ax1[1].imshow(filt1, cmap="gray", vmax=1, vmin=0)
ax1[1].set_title("Filtro")
ax1[2].imshow(img_filt1, cmap="gray", vmax=255, vmin=0)
ax1[2].set_title("Imagen filtrada")
mag_img1, ph_img1, sh_img1 = uf.get_dft(imagen1)

params_id, is_lowpass = {"sigma":SIGMA}, False
filt2 = uf.gaussian_filter(imagen1.shape, params_id, is_lowpass)
img_filt2 = uf.apply_filter(imagen1, ftype, params_id, is_lowpass)

fig2, ax2 = plt.subplots(1, 3, layout="constrained")
fig2.suptitle("Ej: 3 Filtros", fontsize=16)
ax2[0].imshow(imagen1, cmap="gray", vmax=255, vmin=0)
ax2[0].set_title("Imagen 1")
ax2[1].imshow(filt2, cmap="gray", vmax=1, vmin=0)
ax2[1].set_title("Filtro")
ax2[2].imshow(img_filt2, cmap="gray", vmax=255, vmin=0)
ax2[2].set_title("Imagen filtrada")

plt.show()
