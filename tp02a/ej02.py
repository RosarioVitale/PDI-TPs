"""
Ejercicio 2: Transformaciones no lineales
1. Implemente la transformación logarı́tmica s=clog(1+r) y la transformación
de potencia s=rgamma (c=1).
2. Realice el procesado sobre la imagen 'rmn.jpg', utilizando los dos procesos
por separado.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("../")
from utils import operations as ut
from cvui import cvui

IMAGE_DIR = "../imgs/"
IMAGE_FILE = "rmn.jpg"
img_original = cv2.imread(f"{IMAGE_DIR}{IMAGE_FILE}", cv2.IMREAD_GRAYSCALE)
HEIGHT, WIDTH = img_original.shape

# Crear ventana 1. La idea es que se muestren 3 imagenes:
# 1. Imagen original
# 2. Transformacion logaritmica
# 3. Transformacion de potencia
# Ademas, se muestra el titulo de cada imagen en la parte superior y se 
# definen los margenes
MARGIN = 10
TEXT_HEIGHT = 20
img_win_1 = np.zeros((HEIGHT+2*MARGIN+TEXT_HEIGHT, 
                      3*WIDTH+4*MARGIN, # |img|img|img| 
                      3), np.uint8)

# Crear ventana 2. La idea es un panel de control para las transformaciones
# Primero el slider de c para la transformacion logaritmica, luego los sliders 
# de c y gamma para la transformacion de potencia
SLIDER_HEIGHT = 20
SLIDER_WIDTH = 300
TEXT_WIDTH = 20
img_win_2 = np.zeros((4*MARGIN+3*SLIDER_HEIGHT+6*TEXT_HEIGHT, 
                      SLIDER_WIDTH+TEXT_WIDTH+2*MARGIN, 
                      3), np.uint8)

# Crear ventanas con OpenCV
WINDOW_NAME_1 = "Imagen original y transformaciones"
WINDOW_NAME_2 = "Panel de control"
WINDOW_NAME_3 = "Transformaciones"
cv2.namedWindow(WINDOW_NAME_1)
cv2.namedWindow(WINDOW_NAME_2)
# Inicializar ventana principal con cvui
cvui.init(WINDOW_NAME_1)
# Indicarle a cvui que debe seguir el mouse en las demas ventanas
cvui.watch(WINDOW_NAME_2)
cvui.watch(WINDOW_NAME_3)


# Inicializar variables de control
c_log = [1.0]
c_pow = [1.0]
gamma = [1.0]

# Graficar transformaciones con matplotlib
Rvals = np.arange(0, 256)

fig, ax = plt.subplots(1, 2)

def plot_transforms(Rvals, c_log, c_pow, gamma, fig=fig, ax=ax):
    ax[0].clear()
    ax[1].clear()
    log_vals = ut.computeLog(Rvals, c_log[0])
    pow_vals = ut.computePow(Rvals, gamma[0], c_pow[0])
    ax[0].plot(Rvals, log_vals)
    ax[0].set_title("Transformacion logaritmica")
    ax[1].plot(Rvals, pow_vals)
    ax[1].set_title("Transformacion de potencia")
    fig.canvas.draw()
    return np.array(fig.canvas.renderer._renderer)

while True:
    # Mapeo de la imagen original
    img_log = ut.computeLog(img_original, c=c_log[0])
    img_pow = ut.computePow(img_original, gamma=gamma[0], c=c_pow[0])
    # Graficar transformaciones
    fig_transforms = plot_transforms(Rvals, c_log, c_pow, gamma)

    cvui.context(WINDOW_NAME_1)
    img_win_1[:] = (49, 52, 49) # Refrescar ventana 1
    cvui.text(img_win_1, MARGIN, MARGIN, "Imagen original", 0.5, 0xFFFFFF)
    cvui.text(img_win_1, 2*MARGIN+WIDTH, MARGIN, "Transformacion logaritmica", 
              0.5, 0xFFFFFF)
    cvui.text(img_win_1, 3*MARGIN+2*WIDTH, MARGIN, "Transformacion de potencia", 
              0.5, 0xFFFFFF)
    cvui.image(img_win_1, MARGIN, MARGIN+TEXT_HEIGHT, 
               cv2.cvtColor(img_original, cv2.COLOR_GRAY2BGR))
    cvui.image(img_win_1, 2*MARGIN+WIDTH, MARGIN+TEXT_HEIGHT, 
               cv2.cvtColor(img_log, cv2.COLOR_GRAY2BGR))
    cvui.image(img_win_1, 3*MARGIN+2*WIDTH, MARGIN+TEXT_HEIGHT, 
               cv2.cvtColor(img_pow, cv2.COLOR_GRAY2BGR))
    cvui.update()
    cv2.imshow(WINDOW_NAME_1, img_win_1)

    cvui.context(WINDOW_NAME_2)
    img_win_2[:] = (49, 52, 49) # Refrescar ventana 2
    cvui.text(img_win_2, MARGIN, MARGIN, "Transformacion logaritmica", 0.5, 
              0xFFFFFF)
    cvui.text(img_win_2, MARGIN, MARGIN+2*TEXT_HEIGHT, "C: ", 0.5, 0xFFFFFF)
    cvui.trackbar(img_win_2, MARGIN+TEXT_WIDTH, MARGIN+TEXT_HEIGHT, 
                  SLIDER_WIDTH, c_log, 0.0, 3.0, 50, "%.2f", 
                  cvui.TRACKBAR_HIDE_LABELS)
    cvui.text(img_win_2, MARGIN, 2*MARGIN+3*TEXT_HEIGHT+SLIDER_HEIGHT, 
              "Transformacion de potencia", 0.5, 0xFFFFFF)
    cvui.text(img_win_2, MARGIN, 3*MARGIN+5*TEXT_HEIGHT+SLIDER_HEIGHT, "C: ", 
              0.5, 0xFFFFFF)
    cvui.trackbar(img_win_2, MARGIN+TEXT_WIDTH, 
                  3*MARGIN+4*TEXT_HEIGHT+SLIDER_HEIGHT, SLIDER_WIDTH, c_pow, 
                  0.0, 3.0, 50, "%.2f", cvui.TRACKBAR_HIDE_LABELS)
    cvui.text(img_win_2, MARGIN, 3*MARGIN+6*TEXT_HEIGHT+2*SLIDER_HEIGHT, 
              "Gm: ", 0.5, 0xFFFFFF)
    cvui.trackbar(img_win_2, MARGIN+TEXT_WIDTH, 
                  3*MARGIN+5*TEXT_HEIGHT+2*SLIDER_HEIGHT,
                  SLIDER_WIDTH, gamma, 0.0, 2.0, 50, "%.2f", 
                  cvui.TRACKBAR_HIDE_LABELS)
    cvui.update()
    cv2.imshow(WINDOW_NAME_2, img_win_2)

    cvui.context(WINDOW_NAME_3)
    cv2.imshow(WINDOW_NAME_3, fig_transforms)

    if cv2.waitKey(1) == ord('q'):
        break

