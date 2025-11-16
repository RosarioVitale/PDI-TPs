import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from cvui import cvui
import utils.fourier as uf

IMAGE_DIR = "../imgs/"
IMAGE_FILE = "img_degradada.tif"
#IMAGE_FILE = "HeadCT_degradada.tif"
#IMAGE_FILE = "noisy_moon.jpg"
imagen = cv2.imread(f"{IMAGE_DIR}{IMAGE_FILE}", cv2.IMREAD_GRAYSCALE)

fmag,_,_ = uf.get_dft(imagen)

X, Y = np.where(fmag>215)
cfmag = cv2.cvtColor(fmag, cv2.COLOR_GRAY2RGB)
igns=[]
for i in range(len(X)):
    if ((X[i]-fmag.shape[0]//2)**2+(Y[i]-fmag.shape[1]//2)**2)<100:
        igns.append(i)
X=np.delete(X,igns)
Y=np.delete(Y,igns)
#cfmag[X,Y]=[255,0,0]
for i in range(len(X)):
    cv2.circle(cfmag, (Y[i],X[i]), 5, [255,0,0], 1)
D0=np.sqrt((X[0]-fmag.shape[0]//2)**2+(Y[0]-fmag.shape[1]//2)**2)
params = {"W":30, "n":2, "D0":D0}
idealfiltered=uf.apply_band_filter(imagen, uf.IDEAL_FILTER, params, True)
idealreject=uf.ideal_band_filter(imagen.shape, params, True)
butterfiltered=uf.apply_band_filter(imagen, uf.BUTTERWORTH_FILTER, params,
                                    True)
butterreject=uf.butterworth_band_filter(imagen.shape,params,True)
params = {"W":0.5, "D0":D0}
gaussfiltered=uf.apply_band_filter(imagen, uf.GAUSSIAN_FILTER, params, True)
gaussreject=uf.gaussian_band_filter(imagen.shape,params,True)

fig, ax = plt.subplots(3, 3, layout="constrained")
ax[0][0].imshow(imagen, cmap="gray", vmax=255, vmin=0)
ax[0][0].set_title("Imagen")
ax[0][1].imshow(fmag, cmap="gray", vmax=255, vmin=0)
ax[0][1].set_title("Espectro magnitud")
ax[0][2].imshow(cfmag)
ax[0][2].set_title("Puntos marcados")
ax[1][0].imshow(idealreject, cmap="gray", vmax=1, vmin=0)
ax[1][0].set_title("Ideal aplicado")
ax[1][1].imshow(butterreject, cmap="gray", vmax=1, vmin=0)
ax[1][1].set_title("Butterworth aplicado")
ax[1][2].imshow(gaussreject, cmap="gray", vmax=1, vmin=0)
ax[1][2].set_title("Gauss aplicado")
ax[2][0].imshow(idealfiltered, cmap="gray", vmax=255, vmin=0)
ax[2][0].set_title("Filtrada con Ideal")
ax[2][1].imshow(butterfiltered, cmap="gray", vmax=255, vmin=0)
ax[2][1].set_title("Filtrada con Butterworth")
ax[2][2].imshow(gaussfiltered, cmap="gray", vmax=255, vmin=0)
ax[2][2].set_title("Filtrada con Gauss")

plt.show()
