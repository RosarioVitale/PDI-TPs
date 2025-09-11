"""
Ejercicio 1: Manejo de histograma
4. (Opcional): Investigue la ecualización adaptativa de histogramas CLAHE
(Contrast Limited Adaptive Histogram Equalization)
# create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(img)
"""
import cv2
import matplotlib.pyplot as plt


IMAGE_DIR = "../imgs/"
IMAGE_FILE = "clown.jpg"

imagen = cv2.imread(f"{IMAGE_DIR}{IMAGE_FILE}", cv2.IMREAD_GRAYSCALE)
hist = cv2.calcHist([imagen], [0], None, [256], [0, 256])

img_eq = cv2.equalizeHist(imagen)
hist_eq = cv2.calcHist([img_eq], [0], None, [256], [0, 256])

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
img_c = clahe.apply(imagen)
hist_c = cv2.calcHist([img_c], [0], None, [256], [0, 256])

fig, ((ax11, ax12, ax13),
      (ax21, ax22, ax23)) = plt.subplots(2, 3, layout="constrained")
fig.suptitle("CLAHE", fontsize=16)
ax11.imshow(imagen, cmap="gray", vmax=255, vmin=0)
ax11.set_title("Imagen original")
ax12.imshow(img_eq, cmap="gray", vmax=255, vmin=0)
ax12.set_title("Imagen ecualizada")
ax13.imshow(img_c, cmap="gray", vmax=255, vmin=0)
ax13.set_title("Imagen con CLAHE")
ax21.plot(hist)
ax21.set_title("Histograma original")
ax22.plot(hist_eq)
ax22.set_title("Histograma ecualizado")
ax23.plot(hist_c)
ax23.set_title("Histograma con CLAHE")

plt.show()
