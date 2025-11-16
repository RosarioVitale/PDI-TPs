import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("..")
from utils import fourier as uf

IMAGE_DIR = "../imgs/"
IMAGE_FILE1 = "cameraman.tif"
imagen1 = cv2.imread(f"{IMAGE_DIR}{IMAGE_FILE1}", cv2.IMREAD_GRAYSCALE)
R0 = 20
ftype = uf.IDEAL_FILTER

mag_img1, ph_img1, sh_img1 = uf.get_dft(imagen1)
params_id, is_lowpass = {"R0":R0}, True
filt1 = uf.ideal_filter(imagen1.shape, params_id, is_lowpass)
img_filt1 = uf.apply_filter(imagen1, ftype, params_id, is_lowpass)

fig, ax = plt.subplots(2, 3, layout="constrained")
fig.suptitle("Ej: 3 Filtro Ideal", fontsize=16)
ax[0,0].imshow(imagen1, cmap="gray", vmax=255, vmin=0)
ax[0,0].set_title("Imagen 1")
ax[0,1].imshow(filt1, cmap="gray", vmax=1, vmin=0)
ax[0,1].set_title("Filtro Ideal pasa bajos") 
ax[0,2].imshow(img_filt1, cmap="gray", vmax=255, vmin=0)
ax[0,2].set_title("Imagen filtrada con pasa bajos")
mag_img1, ph_img1, sh_img1 = uf.get_dft(imagen1)

params_id, is_lowpass = {"R0":R0}, False
filt2 = uf.ideal_filter(imagen1.shape, params_id, is_lowpass)
img_filt2 = uf.apply_filter(imagen1, ftype, params_id, is_lowpass)

ax[1,0].set_axis_off()
ax[1,1].imshow(filt2, cmap="gray", vmax=1, vmin=0)
ax[1,1].set_title("Filtro Ideal pasa altos")
ax[1,2].imshow(img_filt2, cmap="gray", vmax=255, vmin=0)
ax[1,2].set_title("Imagen filtrada con pasa altos")

plt.show()
