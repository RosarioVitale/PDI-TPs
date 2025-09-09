"""
Ejercicio 4: Trabajos de aplicación
2. Al final del proceso de manufactura de placas madres, de marca ASUS modelo
A7V600, se obtienen dos clases de producto final: A7V600-x y A7V600-SE.
Implemente un algoritmo, que a partir de una imagen, determine que tipo de
placa es. Haga uso de las técnicas de realce apendidas y utilice las imágenes
a7v600-x.gif y a7v600-SE.gif. Adapte el método de forma que contemple el
reconocimiento de imágenes que han sido afectadas por un ruido aleatorio
impulsivo (a7v600-x(RImpulsivo).gif y a7v600-SE(RImpulsivo).gif).
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from utils import utils, operations, noise
from cvui import cvui

# Cargar imagenes de referencia para el reconocimiento
IMAGE_DIR = "../imgs/"
IMAGE_REF1 = "a7v600-X.gif"
IMAGE_REF2 = "a7v600-SE.gif"
img_r1 = utils.load_gif(f"{IMAGE_DIR}{IMAGE_REF1}")
img_r2 = utils.load_gif(f"{IMAGE_DIR}{IMAGE_REF2}")
HEIGHT, WIDTH, _ = img_r1.shape # Todas las imagenes miden lo mismo

mask = cv2.cvtColor(operations.diferencia(img_r1, img_r2), 
                    cv2.COLOR_BGR2GRAY) > 128

def comparar(img, ref, mask):
    img_m = operations.multiplicacion(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 
                                      mask)
    ref_m = operations.multiplicacion(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), 
                                      mask)
    return utils.error_mse(img_m, ref_m)

def clasificar(img, ref1, ref2, mask):
    err1 = comparar(img, ref1, mask)
    err2 = comparar(img, ref2, mask)

    if err1 < err2:
        return "A7V600-X", err1
    else:
        return "A7V600-SE", err2

# Cargar imagenes a reconocer (con y sin ruido impulsivo)
IMAGE_01 = "a7v600-X.gif"
IMAGE_02 = "a7v600-SE.gif"
IMAGE_03 = "a7v600-X(RImpulsivo).gif"
IMAGE_04 = "a7v600-SE(RImpulsivo).gif"
img_01 = utils.load_gif(f"{IMAGE_DIR}{IMAGE_01}")
img_02 = utils.load_gif(f"{IMAGE_DIR}{IMAGE_02}")
img_03 = utils.load_gif(f"{IMAGE_DIR}{IMAGE_03}")
img_04 = utils.load_gif(f"{IMAGE_DIR}{IMAGE_04}")

# Cargar imagenes a ensuciar con mas ruido impulsivo
img_05 = cv2.cvtColor(noise.add_impulsive_noise(
    cv2.cvtColor(img_01, cv2.COLOR_BGR2GRAY), 0.1, 0.1), cv2.COLOR_GRAY2BGR)
img_06 = cv2.cvtColor(noise.add_impulsive_noise(
    cv2.cvtColor(img_02, cv2.COLOR_BGR2GRAY), 0.1, 0.1), cv2.COLOR_GRAY2BGR)

MARGIN = 10
TEXT_HEIGHT = 20
TEXT_WIDTH = 20
TRACKBAR_WIDTH = WIDTH - TEXT_WIDTH
TRACKBAR_HEIGHT = 40
frm = np.zeros((8*MARGIN+2*TEXT_HEIGHT+2*HEIGHT+TRACKBAR_HEIGHT, 
                5*MARGIN+4*WIDTH, 3), 
               np.uint8)
cv2.namedWindow("Placas")
cvui.init("Placas")
N = [0.1]
while (True):
    cvui.context("Placas")
    frm[:] = (49, 52, 49) # Refrescar ventana 2

    cvui.text(frm,  4*MARGIN+3*WIDTH, 6*MARGIN+0*TEXT_HEIGHT+0*HEIGHT,
              "N:", 0.5, 0xFFFFFF)
    if cvui.trackbar(frm, 4*MARGIN+3*WIDTH+TEXT_WIDTH,
                     6*MARGIN+2*TEXT_HEIGHT+2*HEIGHT, TRACKBAR_WIDTH, N, 0, 
                     0.5, 100, "%.2f", cvui.TRACKBAR_HIDE_LABELS):
        img_05 = cv2.cvtColor(noise.add_impulsive_noise(
            cv2.cvtColor(img_01, cv2.COLOR_BGR2GRAY), N[0], N[0]), 
                              cv2.COLOR_GRAY2BGR)
        img_06 = cv2.cvtColor(noise.add_impulsive_noise(
            cv2.cvtColor(img_02, cv2.COLOR_BGR2GRAY), N[0], N[0]), 
                              cv2.COLOR_GRAY2BGR)

    # Calcular errores y clases
    cls_01, err_01 = clasificar(img_01, img_r1, img_r2, mask)
    cls_02, err_02 = clasificar(img_02, img_r1, img_r2, mask)
    cls_03, err_03 = clasificar(img_03, img_r1, img_r2, mask)
    cls_04, err_04 = clasificar(img_04, img_r1, img_r2, mask)
    cls_05, err_05 = clasificar(img_05, img_r1, img_r2, mask)
    cls_06, err_06 = clasificar(img_06, img_r1, img_r2, mask)

    # Mostrar las imagenes de referencia
    cvui.text(frm,  1*MARGIN+0*WIDTH, 1*MARGIN+0*TEXT_HEIGHT+0*HEIGHT,
              "A7V600-X (referencia)", 0.5, 0xFFFFFF)
    cvui.image(frm, 1*MARGIN+0*WIDTH, 2*MARGIN+1*TEXT_HEIGHT+0*HEIGHT, img_r1)
    cvui.text(frm,  1*MARGIN+0*WIDTH, 4*MARGIN+1*TEXT_HEIGHT+1*HEIGHT,
              "A7V600-SE (referencia)", 0.5, 0xFFFFFF)
    cvui.image(frm, 1*MARGIN+0*WIDTH, 5*MARGIN+2*TEXT_HEIGHT+1*HEIGHT, img_r2)

    # Imagenes sin modificar
    cvui.text(frm,  2*MARGIN+1*WIDTH, 1*MARGIN+0*TEXT_HEIGHT+0*HEIGHT,
              "A7V600-X", 0.5, 0xFFFFFF)
    cvui.image(frm, 2*MARGIN+1*WIDTH, 2*MARGIN+1*TEXT_HEIGHT+0*HEIGHT, img_01)
    cvui.text(frm,  2*MARGIN+1*WIDTH, 2*MARGIN+1*TEXT_HEIGHT+0*HEIGHT,
              f"Clase: {cls_01} (Err: {err_01:.2f})", 0.5, 0xFF0000)
    cvui.text(frm,  2*MARGIN+1*WIDTH, 4*MARGIN+1*TEXT_HEIGHT+1*HEIGHT,
              "A7V600-SE", 0.5, 0xFFFFFF)
    cvui.image(frm, 2*MARGIN+1*WIDTH, 5*MARGIN+2*TEXT_HEIGHT+1*HEIGHT, img_02)
    cvui.text(frm,  2*MARGIN+1*WIDTH, 5*MARGIN+2*TEXT_HEIGHT+1*HEIGHT,
              f"Clase: {cls_02} (Err: {err_02:.2f})", 0.5, 0xFF0000)

    # Imagenes con ruido impulsivo fijo
    cvui.text(frm,  3*MARGIN+2*WIDTH, 1*MARGIN+0*TEXT_HEIGHT+0*HEIGHT,
              "A7V600-X (ruido impulsivo)", 0.5, 0xFFFFFF)
    cvui.image(frm, 3*MARGIN+2*WIDTH, 2*MARGIN+1*TEXT_HEIGHT+0*HEIGHT, img_03)
    cvui.text(frm,  3*MARGIN+2*WIDTH, 2*MARGIN+1*TEXT_HEIGHT+0*HEIGHT,
              f"Clase: {cls_03} (Err: {err_03:.2f})", 0.5, 0xFF0000)
    cvui.text(frm,  3*MARGIN+2*WIDTH, 4*MARGIN+1*TEXT_HEIGHT+1*HEIGHT,
              "A7V600-SE (ruido impulsivo)", 0.5, 0xFFFFFF)
    cvui.image(frm, 3*MARGIN+2*WIDTH, 5*MARGIN+2*TEXT_HEIGHT+1*HEIGHT, img_04)
    cvui.text(frm,  3*MARGIN+2*WIDTH, 5*MARGIN+2*TEXT_HEIGHT+1*HEIGHT,
              f"Clase: {cls_04} (Err: {err_04:.2f})", 0.5, 0xFF0000)

    # Imagenes con ruido impulsivo variable
    cvui.text(frm,  4*MARGIN+3*WIDTH, 1*MARGIN+0*TEXT_HEIGHT+0*HEIGHT,
              f"A7V600-X (ruido impulsivo {N[0]:.2f})", 0.5, 0xFFFFFF)
    cvui.image(frm, 4*MARGIN+3*WIDTH, 2*MARGIN+1*TEXT_HEIGHT+0*HEIGHT, img_05)
    cvui.text(frm,  4*MARGIN+3*WIDTH, 2*MARGIN+1*TEXT_HEIGHT+0*HEIGHT,
              f"Clase: {cls_05} (Err: {err_05:.2f})", 0.5, 0xFF0000)
    cvui.text(frm,  4*MARGIN+3*WIDTH, 4*MARGIN+1*TEXT_HEIGHT+1*HEIGHT,
              f"A7V600-SE (ruido impulsivo {N[0]:.2f})", 0.5, 0xFFFFFF)
    cvui.image(frm, 4*MARGIN+3*WIDTH, 5*MARGIN+2*TEXT_HEIGHT+1*HEIGHT, img_06)
    cvui.text(frm,  4*MARGIN+3*WIDTH, 5*MARGIN+2*TEXT_HEIGHT+1*HEIGHT,
              f"Clase: {cls_06} (Err: {err_06:.2f})", 0.5, 0xFF0000)

    cv2.imshow("Placas", frm)
    key = cv2.waitKey(20)
    if key == 27 or key == ord("q"):
        break

