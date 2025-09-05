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
print(img_win_1.shape, img_original.shape)

# Crear ventana 2. La idea es que se muestre la LUT. Las dimensiones de la 
# ventana deben permitir determinar la LUT a partir de la interaccion del 
# usuario. Ademas, se muestra el titulo de la ventana en la parte superior,
# se definen los margenes y se agrega un boton para resetear LUT
px_per_unit = 2
button_height = 20
img_win_2 = np.zeros((3*margin+px_per_unit*255+text_height+button_height, 
                      2*margin+px_per_unit*255, 3), np.uint8)
lut = 255*np.ones((px_per_unit*255, px_per_unit*255, 3), np.uint8)

# Crear ventanas con OpenCV
WINDOW_NAME_1 = "Imagen original, mapeo y negativo"
WINDOW_NAME_2 = "LUT"
cv2.namedWindow(WINDOW_NAME_1)
cv2.namedWindow(WINDOW_NAME_2)
# Inicializar ventana principal con cvui
cvui.init(WINDOW_NAME_1)
# Indicarle a cvui que debe seguir el mouse en la segunda ventana
cvui.watch(WINDOW_NAME_2)

img_neg = ut.applyNegative(img_original)
a = [1]
c = [0]
Rlims = [(0, 255)]
img_lut = ut.applyLUT(img_original, a, c, Rlims)

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
    cvui.text(img_win_2, margin, margin, "LUT", 0.5, 0xFFFFFF)
    # Boton para resetear
    if cvui.button(img_win_2, margin, 2*margin+px_per_unit*255+text_height, 
                   "Reset LUT"):
        a = [1]
        c = [0]
        Rlims = [(0, 255)]
        img_lut = ut.applyLUT(img_original, a, c, Rlims)
    cvui.image(img_win_2, margin, margin+text_height, lut)

    # TODO: Actualizar LUT

    cvui.update()
    cv2.imshow(WINDOW_NAME_2, img_win_2)

    if cv2.waitKey(1) == ord("q"):
        break
