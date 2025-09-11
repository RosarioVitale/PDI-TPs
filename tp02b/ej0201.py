"""
Ejercicio 2: Trabajos de aplicación
1. En la imagen cuadros.tif se observa un conjunto de cuadros negros sobre un
fondo casi uniforme. Utilice ecualización local del histograma para revelar los
detalles ocultos en la imagen y compare los resultados con los obtenidos con
ecualización global.
Ayuda: el tamaño de ventana y su localización es clave para realizar la 
ecualización local.
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np


IMAGE_DIR = "../imgs/"
IMAGE_FILE = "cuadros.tif"

imagen = cv2.imread(f"{IMAGE_DIR}{IMAGE_FILE}", cv2.IMREAD_GRAYSCALE)
hist = cv2.calcHist([imagen], [0], None, [256], [0, 256])

img_eqg = cv2.equalizeHist(imagen)
hist_eqg = cv2.calcHist([img_eqg], [0], None, [256], [0, 256])

#* Esto hace lo siguiente: tomamos los indices de las zonas que queremos
#* ecualizar, extraemos esos pixeles, los ecualizamos y (en una imagen nueva)
#* reacomodamos los pixeles de la zona ecualizada
inds = imagen<100
img_win = imagen[inds]
imgw_eq = cv2.equalizeHist(img_win)
img_eql = np.copy(imagen)
img_eql[inds] = imgw_eq[:,0]
hist_eql = cv2.calcHist([img_eql], [0], None, [256], [0, 256])

fig, ((ax11, ax12, ax13),
      (ax21, ax22, ax23)) = plt.subplots(2, 3, layout="constrained")
fig.suptitle("Realzar cuadros negros", fontsize=16)
ax11.imshow(imagen, cmap="gray", vmax=255, vmin=0)
ax11.set_title("Imagen original")
ax12.imshow(img_eqg, cmap="gray", vmax=255, vmin=0)
ax12.set_title("Imagen ec. global")
ax13.imshow(img_eql, cmap="gray", vmax=255, vmin=0)
ax13.set_title("Imagen eq. local")
ax21.plot(hist)
ax21.set_title("Histograma original")
ax22.plot(hist_eqg)
ax22.set_title("Histograma ec. global")
ax23.plot(hist_eql)
ax23.set_title("Histograma ec. local")

plt.show()
