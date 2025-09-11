"""
Ejercicio 4: Trabajos de aplicación
3. En una fábrica de medicamentos se desea implementar un sistema para la
inspección visual automática de blisters en la lı́nea de empaquetado. La 
adquisición de la imagen se realiza en escala de grises mediante una cámara
CCD fija y bajo condiciones controladas de iluminación, escala y enfoque. El
objetivo consiste en determinar en cada instante si el blister que está siendo
analizado se encuentra incompleto, en cuyo caso la región correspondiente a
la pı́ldora faltante presenta una intensidad similar al fondo. Escriba una 
función que reciba como parámetro la imagen del blister a analizar y devuelva
un mensaje indicando si el mismo contiene o no la totalidad de las pı́ldoras. 
En caso de estar incompleto, indique la posición (x,y) de las pı́ldoras
faltantes. Verifique el funcionamiento con las imágenes blister_completo.jpg y
blister_incompleto.jpg.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from utils import utils, operations
import os

# TODO: Definir la posicion de las pildoras a partir de una imagen de blister
# completo automaticamente
DEFAULT_POS = [[50,50], [50,100], [50,150], [50,200], [50,250], [100,50],
               [100,100], [100,150], [100,200], [100,250]]

def detect_missing_pills(image, bg_th=100, positions=DEFAULT_POS):
    """TODO: Docstring for detect_missing_pills.

    Parameters:
        image: TODO
        bg_th: Umbral para el fondo. Todo lo que sea menor a este numero se
        considera fondo. Valor por defecto=100.
    Returns:
        TODO

    """
    mask = image < bg_th
    image=cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    missing_pills = []
    for pos in positions:
        if mask[pos[0], pos[1]] != 0:
            missing_pills.append(pos)
            cv2.rectangle(image, (pos[1]-15, pos[0]-15),
                          (pos[1]+15, pos[0]+15), (0, 0, 255), 1)
    return image


# Cargar imagen de referencia
IMAGE_DIR = "blisters/"
filenames = [f for f in os.listdir(IMAGE_DIR) if "blister_" in f]
RES_DIR = "blisters_results/"
if not os.path.exists(RES_DIR):
    os.makedirs(RES_DIR)

for filename in filenames:
    image = cv2.imread(f"{IMAGE_DIR}{filename}", cv2.IMREAD_GRAYSCALE)
    image = detect_missing_pills(image)
    cv2.imwrite(f"{RES_DIR}{filename}", image)
