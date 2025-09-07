"""
Ejercicio 1: Transformaciones lineales de una imagen
Conociendo la ecuación general de una transformación lineal:
    s = ar + c
con r: valor de entrada, a: factor de ganancia y c: offset, realice los 
siguientes ejercicios:
    1. Implemente una LUT del mapeo entre la entrada y la salida.
    2. Pruebe la rutina con diferentes juegos de coeficientes a y c, sobre
    diversas imágenes, y muestre en una misma ventana la imagen original, el 
    mapeo aplicado y la imagen obtenida.
    3. Implemente el negativo de la imagen de entrada.
    4. Genere diversas LUT con estiramientos y compresiones lineales por tramos 
    de la entrada, y pruebe los resultados sobre diversas imágenes.
    5. (Opcional): genere una imagen binaria de 256x256 que simule los ejes
    cartesianos de la transformación r-s, marcando la lı́nea identidad. Capture
    dos puntos que marquen el final de los segmentos, y genere la LUT con una
    transformación que parta del origen, pase por los puntos marcados, y
    finalice en el punto (256,256). Pruebe la rutina con diversas imágenes.
    Recomendación: utilizar matplotlib (https://matplotlib.org/) con subplots, 
    y para manejos de eventos, ver: 
    https://matplotlib.org/users/event_handling.html
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import operations as ut

import sys
cvui_dir = "../cvui"
if cvui_dir not in sys.path:
    sys.path.append(cvui_dir)
import cvui

import bisect

IMAGE_DIR = "../imgs/"
IMAGE_FILE = "clown.jpg"
img_original = cv2.imread(f"{IMAGE_DIR}{IMAGE_FILE}", cv2.IMREAD_GRAYSCALE)
HEIGHT, WIDTH = img_original.shape

# Crear ventana 1. La idea es que se muestren 3 imagenes:
# 1. Imagen original
# 2. Mapeo aplicado
# 3. Negativo
# Ademas, se muestra el titulo de cada imagen en la parte superior y se 
# definen los margenes
margin = 10
text_height = 20
img_win_1 = np.zeros((HEIGHT+2*margin+text_height,
                      3*WIDTH+4*margin, # |img|img|img| 
                      3), np.uint8)

# Crear ventana 2. La idea es que se muestre la LUT. Las dimensiones de la 
# ventana deben permitir determinar la LUT a partir de la interaccion del 
# usuario. Ademas, se muestra el titulo de la ventana en la parte superior,
# se definen los margenes y se agrega un boton para resetear LUT
px_per_unit = 1
button_height = 20
img_win_2 = np.zeros((2*margin+px_per_unit*256+text_height+button_height, 
                      px_per_unit*256, 3), np.uint8)

# Crear ventanas con OpenCV
WINDOW_NAME_1 = "Imagen original, mapeo y negativo"
WINDOW_NAME_2 = "LUT"
cv2.namedWindow(WINDOW_NAME_1)
cv2.namedWindow(WINDOW_NAME_2)
# Inicializar ventana principal con cvui
cvui.init(WINDOW_NAME_1)
# Indicarle a cvui que debe seguir el mouse en la segunda ventana
cvui.watch(WINDOW_NAME_2)

# Aplicar transformaciones
img_neg = ut.applyNegative(img_original)
points = []
a, c, Rlims = ut.hacerTramos(points)
img_lut = ut.applyLUT(img_original, a, c, Rlims)

# Variables para dibujar LUT
lut = 255*np.ones((px_per_unit*256, px_per_unit*256, 3), np.uint8)
#print("LUT:", lut.shape)
lut_x_min, lut_y_min = 0, 0
lut_width, lut_height = 256, 256
R_points = np.arange(Rlims[0][0], Rlims[0][1]+px_per_unit, px_per_unit)
lut_points = ut.computeLUT(R_points, a[0], c[0])

while (True):
    cvui.context(WINDOW_NAME_1)
    img_win_1[:] = (49, 52, 49) # Refrescar ventana 1
    cvui.text(img_win_1, margin, margin, "Imagen original", 0.5, 0xFFFFFF)
    cvui.text(img_win_1, 2*margin+WIDTH, margin, "Imagen resultante", 0.5, 
              0xFFFFFF)
    cvui.text(img_win_1, 3*margin+2*WIDTH, margin, "Imagen negativa", 0.5, 
              0xFFFFFF)
    cvui.image(img_win_1, margin, margin+text_height, 
               cv2.cvtColor(img_original, cv2.COLOR_GRAY2BGR))
    cvui.image(img_win_1, 2*margin+WIDTH, margin+text_height,
               cv2.cvtColor(img_lut, cv2.COLOR_GRAY2BGR))
    cvui.image(img_win_1, 3*margin+2*WIDTH, margin+text_height,
               cv2.cvtColor(img_neg, cv2.COLOR_GRAY2BGR))
    cvui.update()
    cv2.imshow(WINDOW_NAME_1, img_win_1)

    cvui.context(WINDOW_NAME_2)
    img_win_2[:] = (49, 52, 49) # Refrescar ventana 2
    # Boton para resetear
    if cvui.button(img_win_2, 0, margin+lut.shape[0], "Reset LUT"):
        points = []
        a, c, Rlims = ut.hacerTramos(points)
        R_points = np.arange(Rlims[0][0], Rlims[0][1]+px_per_unit, px_per_unit)
        lut_points = ut.computeLUT(R_points, a[0], c[0])
        img_lut = ut.applyLUT(img_original, a, c, Rlims)
    # TODO: Actualizar LUT
    if cvui.mouse(cvui.LEFT_BUTTON, cvui.CLICK):
        x, y = cvui.mouse().x, cvui.mouse().y
        if (y<256):
            #print("Click:", x, y, "- Inserting point:", x, 255-y)
            #print("Points before: ", points)
            if len(points) == 0:
                points = [(x, 255-y)]
            else:
                bisect.insort(points, (x, 255-y))
            #print("Points after: ", points)
            a, c, Rlims = ut.hacerTramos(points)
            i0 = 0
            for i in range(0, len(Rlims)):
                R_points = np.arange(Rlims[i][0], Rlims[i][1], px_per_unit)
                lut_points[i0:i0+len(R_points)] = ut.computeLUT(R_points, 
                                                                a[i], c[i])
                i0 += len(R_points)
            img_lut = ut.applyLUT(img_original, a, c, Rlims)

    # TODO: Actualizar LUT
    cvui.image(img_win_2, 0, 0, lut)
    cvui.sparkline(img_win_2, lut_points, 0, 0,
                   lut_width*px_per_unit, lut_height*px_per_unit, 0xFF0000)
    cvui.update()
    cv2.imshow(WINDOW_NAME_2, img_win_2)

    if cv2.waitKey(1) == ord("q"):
        break
