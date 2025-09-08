"""
Ejercicio 3: Operaciones aritméticas
1. Implemente una función que realice las siguientes operaciones aritméticas
sobre dos imágenes que sean pasadas como parámetros:
    a) Suma. Normalice el resultado por el número de imágenes.
    b) Diferencia. Aplique las dos funciones de reescalado usadas tı́picamente
    para evitar el desborde de rango (sumar 255 y dividir por 2, o restar el
    mı́nimo y escalar a 255).
    c) Multiplicación. En esta operación la segunda imagen deberá ser una
    máscara binaria, muy utilizada para la extracción de la región de interés
    (ROI) de una imagen.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("../")
from utils import operations as ut
from cvui import cvui

IMAGE_DIR = "../imgs/"
IMAGE_FILE_1 = "clown.jpg"
IMAGE_FILE_2 = "rmn.jpg"
img_1 = cv2.imread(f"{IMAGE_DIR}{IMAGE_FILE_1}", cv2.IMREAD_GRAYSCALE)
img_2 = cv2.imread(f"{IMAGE_DIR}{IMAGE_FILE_2}", cv2.IMREAD_GRAYSCALE)
mask = img_2 > 128

# a) Suma
sum_img = ut.suma(img_1, img_2)
# b) Diferencia
diff_img1 = ut.diferencia(img_1, img_2, "sum")
diff_img2 = ut.diferencia(img_1, img_2, "res")
diff_img3 = ut.diferencia(img_1, img_2, "no")
# c) Multiplicación
mult_img = ut.multiplicacion(img_1, mask)
# Blending
blending_img = ut.blending(img_1, img_2, 0.5)

fig, ax = plt.subplots(3, 3, tight_layout=True)
ax[0, 0].imshow(img_1, cmap="gray")
ax[0, 0].set_title("Imagen 1")
ax[0, 1].imshow(img_2, cmap="gray")
ax[0, 1].set_title("Imagen 2")
ax[0, 2].imshow(mask, cmap="gray")
ax[0, 2].set_title("Mascara")

ax[1, 0].imshow(sum_img, cmap="gray")
ax[1, 0].set_title("Suma")
ax[1, 1].imshow(blending_img, cmap="gray")
ax[1, 1].set_title("Blending 0.5")
ax[1, 2].imshow(mult_img, cmap="gray")
ax[1, 2].set_title("Multiplicación")

ax[2, 0].imshow(diff_img1, cmap="gray")
ax[2, 0].set_title("Diferencia (sum)")
ax[2, 1].imshow(diff_img2, cmap="gray")
ax[2, 1].set_title("Diferencia (res)")
ax[2, 2].imshow(diff_img3, cmap="gray")
ax[2, 2].set_title("Diferencia (no)")
plt.show()
