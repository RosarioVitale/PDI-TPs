"""
Ejercicio 4: Trabajos de aplicación
1. Utilizando las técnicas aprendidas, descubra que objetos no están
perceptibles en la imagen earth.bmp y realce la imagen de forma que los 
objetos se vuelvan visibles con buen contraste sin realizar modificaciones 
sustanciales en el resto de la imagen.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from utils import operations as ut
from cvui import cvui
import bisect

IMAGE_DIR = "../imgs/"
IMAGE_FILE = "earth.bmp"
img_original = cv2.imread(f"{IMAGE_DIR}{IMAGE_FILE}", cv2.IMREAD_GRAYSCALE)
HEIGHT, WIDTH = img_original.shape

# Crear ventana 1. La idea es que se muestren 2 imagenes:
# 1. Imagen original
# 2. Mapeo aplicado
# Ademas, se muestra el titulo de cada imagen en la parte superior y se 
# definen los margenes
margin = 10
text_height = 20
img_win_1 = np.zeros((HEIGHT+2*margin+text_height,
                      2*WIDTH+3*margin, # |img|img|
                      3), np.uint8)

# Graficar transformaciones con matplotlib
def plot_transforms(Rvals, current_transform, lut_points, c_log, c_pow, gamma, 
                    fig, ax):
    ax.clear()
    if current_transform == "LUT":
        ax.plot(Rvals, lut_points)
        ax.set_title("Mapeo aplicado")
    elif current_transform == "LOG":
        log_vals = ut.computeLog(Rvals, c_log[0])
        ax.plot(Rvals, log_vals)
        ax.set_title("Transformacion logaritmica")
    elif current_transform == "POW":
        pow_vals = ut.computePow(Rvals, gamma[0], c_pow[0])
        ax.plot(Rvals, pow_vals)
        ax.set_title("Transformacion de potencia")
    fig.canvas.draw()
    return np.array(fig.canvas.renderer._renderer)

# Crear ventana 2. La idea es que se muestre la LUT. Las dimensiones de la 
# ventana deben permitir determinar la LUT a partir de la interaccion del 
# usuario. Ademas, se muestra el titulo de la ventana en la parte superior,
# se definen los margenes y se agrega un boton para resetear LUT
current_transform = "LUT" # opciones: "LUT", "LOG", "POW"
button_height = 20
button_width = 100
trackbar_width = 200
trackbar_height = 20
text_width = 100
Rvals = np.arange(0, 256)
fig, ax = plt.subplots(1, 1)
c_log, c_pow, gamma = [1.0], [1.0], [1.0]

# Crear ventanas con OpenCV
WINDOW_NAME_1 = "Imagen original y transformada"
WINDOW_NAME_2 = "Transformaciones"
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
img_trn = ut.applyLUT(img_original, a, c, Rlims)

# Variables para dibujar LUT
lut = 255*np.ones((256, 256, 3), np.uint8)
lut_x_min, lut_y_min = 0, 0
lut_width, lut_height = 256, 256
R_points = np.arange(Rlims[0][0], Rlims[0][1]+1, 1)
lut_points = ut.computeLUT(R_points, a[0], c[0])
idt_points = ut.computeLUT(R_points, 1, 0)
img_plt = plot_transforms(Rvals, current_transform, lut_points, c_log, c_pow, 
                          gamma, fig, ax)
img_win_2 = np.zeros((3*margin+np.max([256, img_plt.shape[0]])+text_height+
                      button_height+3*trackbar_height, 
                      3*margin+256+img_plt.shape[1], 
                      3), np.uint8)

while (True):
    cvui.context(WINDOW_NAME_1)
    img_win_1[:] = (49, 52, 49) # Refrescar ventana 1
    cvui.text(img_win_1, margin, margin, "Imagen original", 0.5, 0xFFFFFF)
    cvui.text(img_win_1, 2*margin+WIDTH, margin, "Imagen resultante", 0.5, 
              0xFFFFFF)
    cvui.image(img_win_1, margin, margin+text_height, 
               cv2.cvtColor(img_original, cv2.COLOR_GRAY2BGR))
    cvui.image(img_win_1, 2*margin+WIDTH, margin+text_height,
               cv2.cvtColor(img_trn, cv2.COLOR_GRAY2BGR))
    cvui.update()
    cv2.imshow(WINDOW_NAME_1, img_win_1)

    cvui.context(WINDOW_NAME_2)
    img_win_2[:] = (49, 52, 49) # Refrescar ventana 2
    # Botones para resetear
    cvui.text(img_win_2, margin, 3*margin+lut.shape[0], "Reset buttons", 0.5, 
              0xFFFFFF)
    if cvui.button(img_win_2, margin, 3*margin+lut.shape[0]+text_height, 
                   "Reset LUT"):
        current_transform = "LUT"
        points = []
        a, c, Rlims = ut.hacerTramos(points)
        R_points = np.arange(Rlims[0][0], Rlims[0][1]+1, 1)
        lut_points = ut.computeLUT(R_points, a[0], c[0])
        img_trn = ut.applyLUT(img_original, a, c, Rlims)
    if cvui.button(img_win_2, margin, 5*margin+lut.shape[0]+text_height+
                   button_height, "Reset LOG"):
        current_transform = "LOG"
        c_log = [1.0]
        img_trn = ut.computeLog(img_original, c_log[0])
    if cvui.button(img_win_2, margin, 7*margin+lut.shape[0]+text_height+
                   2*button_height,
                   "Reset POW"):
        current_transform = "POW"
        c_pow, gamma = [1.0], [1.0]
        img_trn = ut.computePow(img_original, gamma[0], c_pow[0])

    # Seleccionar puntos de la LUT para modificarla
    if cvui.mouse(cvui.LEFT_BUTTON, cvui.CLICK):
        x, y = cvui.mouse().x, cvui.mouse().y
        if (y<(256+margin) and x<(256+margin) and x>=margin and y>=margin):
            current_transform = "LUT"
            if len(points) == 0:
                points = [(x-margin, 255-y+margin)]
            else:
                bisect.insort(points, (x-margin, 255-y+margin))
            a, c, Rlims = ut.hacerTramos(points)
            i0 = 0
            for i in range(0, len(Rlims)):
                R_points = np.arange(Rlims[i][0], Rlims[i][1], 1)
                lut_points[i0:i0+len(R_points)] = ut.computeLUT(R_points, 
                                                                a[i], c[i])
                i0 += len(R_points)
            img_trn = ut.applyLUT(img_original, a, c, Rlims)

    # Etiquetar trackbars
    cvui.text(img_win_2, 2*margin+256, 2*margin+img_plt.shape[0],
              "C LOG", 0.5, 0xFFFFFF)
    cvui.text(img_win_2, 2*margin+256, 6*margin+img_plt.shape[0],
              "C POW", 0.5, 0xFFFFFF)
    cvui.text(img_win_2, 2*margin+256, 10*margin+img_plt.shape[0],
              "GAMMA", 0.5, 0xFFFFFF)
    # Seleccionar transformacion al cambiar trackbars
    if cvui.trackbar(img_win_2, 2*margin+256+text_width, 
                     0*margin+img_plt.shape[0],
                     trackbar_width, c_log, 0.0, 2.0, 50, "%.2f", 
                     cvui.TRACKBAR_HIDE_LABELS):
        current_transform = "LOG"
        img_trn = ut.computeLog(img_original, c_log[0])
    if cvui.trackbar(img_win_2, 2*margin+256+text_width, 
                     2*margin+img_plt.shape[0]+trackbar_height, 
                     trackbar_width, c_pow, 0.0, 2.0, 50, "%.2f", 
                     cvui.TRACKBAR_HIDE_LABELS):
        current_transform = "POW"
        img_trn = ut.computePow(img_original, gamma[0], c_pow[0])
    if cvui.trackbar(img_win_2, 2*margin+256+text_width, 
                     4*margin+img_plt.shape[0]+2*trackbar_height,
                     trackbar_width, gamma, 0.0, 2.0, 50, "%.2f", 
                     cvui.TRACKBAR_HIDE_LABELS):
        current_transform = "POW"
        img_trn = ut.computePow(img_original, gamma[0], c_pow[0])
    # Dibujar LUT
    cvui.image(img_win_2, margin, margin, lut)
    cvui.sparkline(img_win_2, idt_points, margin, margin, lut_width, 
                   lut_height, 0x00FF00)
    cvui.sparkline(img_win_2, lut_points, margin, margin, lut_width, 
                   lut_height, 0xFF0000)

    # Dibujar transformaciones
    img_plt = plot_transforms(Rvals, current_transform, lut_points, c_log, 
                              c_pow, gamma, fig, ax)
    cvui.image(img_win_2, 2*margin+256, margin, 
               cv2.cvtColor(img_plt, cv2.COLOR_BGR2RGB))
    cvui.update()
    cv2.imshow(WINDOW_NAME_2, img_win_2)

    if cv2.waitKey(1) == ord("q"):
        break

