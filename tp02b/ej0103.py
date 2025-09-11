"""
Ejercicio 1: Manejo de histograma
Cargue una imagen y realice la ecualización de su histograma.
 img equ = cv2.equalizeHist(img)
Muestre en una misma ventana la imagen original, la versión ecualizada y sus 
respectivos histogramas. Estudie la información suministrada por los 
histogramas. ¿Qué diferencias nota respecto a las definiciones teóricas?
 Repita el análisis para distintas imágenes.
"""
import cv2
import matplotlib.pyplot as plt


IMAGE_DIR = "../imgs/"
IMAGE_FILE = "earth.bmp"
imagen = cv2.imread(f"{IMAGE_DIR}{IMAGE_FILE}", cv2.IMREAD_GRAYSCALE)

hist = cv2.calcHist([imagen], [0], None, [256], [0, 256])
img_eq = cv2.equalizeHist(imagen)
hist_eq = cv2.calcHist([img_eq], [0], None, [256], [0, 256])

fig, ((ax11, ax12), (ax21, ax22)) = plt.subplots(2, 2, layout="constrained")
fig.suptitle("Ecualizacion", fontsize=16)
ax11.imshow(imagen, cmap="gray", vmax=255, vmin=0)
ax11.set_title("Imagen original")
ax12.stem(hist)
ax12.set_title("Histograma original")
ax21.imshow(img_eq, cmap="gray", vmax=255, vmin=0)
ax21.set_title("Imagen ecualizada")
ax22.stem(hist_eq)
ax22.set_title("Histograma ecualizado")

plt.show()
