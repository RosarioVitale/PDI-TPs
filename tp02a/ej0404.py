"""
Ejercicio 4: Trabajos de aplicación
4. (opcional) Implemente una función que permita "esconder" una imagen binaria 
en una imagen de grises sin que ésto sea percibido a simple vista. Luego,
implemente una función que permita extraer la imagen binaria. Analice su
desempeño. [Utilice rodajas del plano de bits]
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from utils import utils, operations
import os

IMG_DIR = "../imgs/"
IMG_1 = "cameraman.tif"
IMG_2 = "clown.jpg"
RES_DIR = "hiden_images/"
if not os.path.exists(RES_DIR):
    os.makedirs(RES_DIR)

img = cv2.imread(f"{IMG_DIR}{IMG_1}", cv2.IMREAD_GRAYSCALE)
secret = cv2.imread(f"{IMG_DIR}{IMG_2}", cv2.IMREAD_GRAYSCALE)
secret = secret > 128


def convert2bitplane(image):
    """Recibe una imagen en grises y devuelve la descompoción en planos de 
    bits.

    Parameters:
        image: imagen en grises

    Returns:
        bit_planes: lista de 8 imagenes binarias (tener en cuenta que el 
        pixel negro es 0 y el blanco es 1)

    """
    bit_planes = []
    for i in range(8): 
        # Extract the i-th bit plane
        bit_plane = (image & (1 << i))/(1 << i)
        bit_planes.append(bit_plane)
    return bit_planes


def convert2gray(bit_planes):
    """Recibe los planos de bits y devuelve la imagen en grises.

    Parameters:
        bit_planes: lista de 8 imagenes binarias. Tener en cuenta que el pixel
        negro es 0 y el blanco es 1

    Returns:
        image: imagen en grises

    """
    image = np.zeros(bit_planes[0].shape, dtype=np.uint8)
    for i in range(8):
        image += np.astype(bit_planes[i] * (2**i), np.uint8)
    return image


def hide_image(image, secret, plane):
    """TODO: Docstring for hide_image.

    Parameters:
        image: TODO
        secret: TODO
        plane: TODO

    Returns:
        TODO

    """
    image = convert2bitplane(image)
    image[plane] = secret
    return convert2gray(image)


def extract_image(image, plane):
    """TODO: Docstring for extract_image.

    Parameters:
        image: TODO

    Returns:
        TODO

    """
    return convert2bitplane(image)[plane]

"""
fig1, axs1 = plt.subplots(3,3)
fig1.tight_layout()
bit_planes = convert2bitplane(img)
axs1[0, 0].imshow(img, cmap="gray")
axs1[0, 0].set_title("Original")
axs1[0, 0].set_axis_off()
axs1[0, 1].imshow(bit_planes[0], cmap="gray")
axs1[0, 1].set_title("Plano 0")
axs1[0, 1].set_axis_off()
axs1[0, 2].imshow(bit_planes[1], cmap="gray")
axs1[0, 2].set_title("Plano 1")
axs1[0, 2].set_axis_off()
axs1[1, 0].imshow(bit_planes[2], cmap="gray")
axs1[1, 0].set_title("Plano 2")
axs1[1, 0].set_axis_off()
axs1[1, 1].imshow(bit_planes[3], cmap="gray")
axs1[1, 1].set_title("Plano 3")
axs1[1, 1].set_axis_off()
axs1[1, 2].imshow(bit_planes[4], cmap="gray")
axs1[1, 2].set_title("Plano 4")
axs1[1, 2].set_axis_off()
axs1[2, 0].imshow(bit_planes[5], cmap="gray")
axs1[2, 0].set_title("Plano 5")
axs1[2, 0].set_axis_off()
axs1[2, 1].imshow(bit_planes[6], cmap="gray")
axs1[2, 1].set_title("Plano 6")
axs1[2, 1].set_axis_off()
axs1[2, 2].imshow(bit_planes[7], cmap="gray")
axs1[2, 2].set_title("Plano 7")
axs1[2, 2].set_axis_off()
"""

fig2, axs2 = plt.subplots(1, 3)
fig2.tight_layout()
fig2.suptitle("Imagen oculta")
axs2[0].imshow(img, cmap="gray")
axs2[0].set_title("Original")
axs2[0].set_axis_off()
axs2[1].imshow(secret, cmap="gray")
axs2[1].set_title("Mascara")
axs2[1].set_axis_off()
axs2[2].imshow(hide_image(img, secret, 4), cmap="gray")
axs2[2].set_title("Imagen oculta")
axs2[2].set_axis_off()

plt.show()
