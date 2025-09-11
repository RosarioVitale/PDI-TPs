"""
Ejercicio 1: Manejo de histograma
1. Cargue y visualice la imágenes patron2.tif y patron.tif (a esta última
utilı́cela a escala de grises). Reflexione acerca de que histograma espera 
obtener para cada una. Obtenga los histogramas y grafı́quelos.

histr = cv2.calcHist(images, channels, mask, histSize, 
                     ranges[,hist[,accumulate]])pyplot.plot(histr) 

Identifique la información suministrada y analı́cela en relación a su 
expectativa.
"""
import cv2
import matplotlib.pyplot as plt


IMAGE_DIR = "../imgs/"
IMAGE_FILE1 = "patron.tif"
IMAGE_FILE2 = "patron2.tif"
imagen1 = cv2.imread(f"{IMAGE_DIR}{IMAGE_FILE1}", cv2.IMREAD_GRAYSCALE)
imagen2 = cv2.imread(f"{IMAGE_DIR}{IMAGE_FILE2}", cv2.IMREAD_GRAYSCALE)

hist1 = cv2.calcHist([imagen1], [0], None, [256], [0, 256])
hist2 = cv2.calcHist([imagen2], [0], None, [256], [0, 256])

fig, ax = plt.subplots(2, 2, layout="constrained")
fig.suptitle("Imagenes e histogramas", fontsize=16)
ax[0][0].imshow(imagen1, cmap="gray", vmax=255, vmin=0)
ax[0][0].set_title("Imagen 1")
ax[0][1].stem(hist1)
ax[0][1].set_title("Histograma 1")  
ax[1][0].imshow(imagen2, cmap="gray", vmax=255, vmin=0)
ax[1][0].set_title("Imagen 2")
ax[1][1].stem(hist2)
ax[1][1].set_title("Histograma 2")
plt.show()
