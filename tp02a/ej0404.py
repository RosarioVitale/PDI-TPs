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
IMG_1 = "cameraman.jpg"
IMG_2 = "clown.jpg"
RES_DIR = "hiden_images/"
if not os.path.exists(RES_DIR):
    os.makedirs(RES_DIR)


def hide_image(image, secret, channel=0):
    """TODO: Docstring for hide_image.

    Parameters:
        image: TODO
        secret: TODO

    Returns:
        TODO

    """
    return


def extract_image(image, channel=0):
    """TODO: Docstring for extract_image.

    Parameters:
        image: TODO

    Returns:
        TODO

    """
    return

