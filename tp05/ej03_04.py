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
mag_img1, _, sh_img1 = uf.get_dft(imagen1)

###############################
#  Filtrado desde el espacio  #
###############################
k = cv2.getGaussianKernel(35, -1)
k /= k[0,0]
filt1 = k * k.T
filt1 /= filt1.sum()
cv2.normalize(filt1, filt1, 0, 1, cv2.NORM_MINMAX)

mag_filt1, _, sh_filt1 = uf.get_dft(filt1)
rsz_sh_filt1=np.zeros_like(sh_img1)
rsz_sh_filt1[:,:,0] = cv2.resize(sh_filt1[:,:,0], imagen1.shape)
rsz_sh_filt1[:,:,1] = cv2.resize(sh_filt1[:,:,1], imagen1.shape)
prod_tdf1 = cv2.mulSpectrums(sh_img1, rsz_sh_filt1, cv2.DFT_ROWS)
img_filt1 = uf.get_idft(prod_tdf1)
mag_prod=cv2.magnitude(prod_tdf1[:,:,0], prod_tdf1[:,:,1])
cv2.log(1+mag_prod,mag_prod)
cv2.normalize(mag_prod, mag_prod, 0, 255, cv2.NORM_MINMAX)
############################
#  Filtrado en frecuencia  #
############################
params_id, is_lowpass = {"sigma":SIGMA}, True
filt_frq1 = uf.gaussian_filter(imagen1.shape, params_id, is_lowpass)
img_filt_frq1 = uf.apply_filter(imagen1, ftype, params_id, is_lowpass)

#############################
#  Figuras para pasa bajos  #
#############################
fig, ax = plt.subplots(3, 3, layout="constrained")
fig.suptitle("Ej: 3 Filtro gaussiano pasa bajos en espacio y frecuencia", 
             fontsize=16)
ax[0][0].imshow(imagen1, cmap="gray", vmax=255, vmin=0)
ax[0][0].set_title("Imagen Original")
ax[1][0].imshow(mag_img1, cmap="gray", vmax=255, vmin=0)
ax[1][0].set_title("TDF imagen")
ax[1][2].imshow(mag_prod, cmap="gray", vmax=255, vmin=0)
ax[1][2].set_title("Producto de TDFs")
ax[0][1].imshow(filt1, cmap="gray", vmax=1, vmin=0)
ax[0][1].set_title("Filtro")
ax[1][1].imshow(mag_filt1, cmap="gray", vmax=255, vmin=0)
ax[1][1].set_title("TDF filtro")
ax[0][2].imshow(img_filt1, cmap="gray", vmax=255, vmin=0)
ax[0][2].set_title("Imagen filtrada")
ax[2][0].set_axis_off()
ax[2][1].imshow(filt_frq1, cmap="gray", vmax=1, vmin=0)
ax[2][1].set_title("Filtro")
ax[2][2].imshow(img_filt_frq1, cmap="gray", vmax=255, vmin=0)
ax[2][2].set_title("Imagen filtrada")

#params_id, is_lowpass = {"sigma":SIGMA}, False
#filt2 = uf.gaussian_filter(imagen1.shape, params_id, is_lowpass)
#img_filt2 = uf.apply_filter(imagen1, ftype, params_id, is_lowpass)
#
#fig2, ax2 = plt.subplots(1, 3, layout="constrained")
#fig2.suptitle("Ej: 3 Filtros", fontsize=16)
#ax2[0].imshow(imagen1, cmap="gray", vmax=255, vmin=0)
#ax2[0].set_title("Imagen 1")
#ax2[1].imshow(filt2, cmap="gray", vmax=1, vmin=0)
#ax2[1].set_title("Filtro")
#ax2[2].imshow(img_filt2, cmap="gray", vmax=255, vmin=0)
#ax2[2].set_title("Imagen filtrada")

plt.show()
