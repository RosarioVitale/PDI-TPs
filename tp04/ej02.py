import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from cvui import cvui
from utils import utils as ut
from utils import histogram as hi


###############################################################################
#                      Constantes para ubicar elementos                       #
###############################################################################
PADX, PADY = 10, 10
SLIDER_WIDTH, SLIDER_HEIGHT = 250, 30
###############################################################################
#                        Funciones para este ejercicio                        #
###############################################################################
def apply_threshold(image, th0, th1, fig, ax):
    global hist_arr
    ax.clear()
    ax.stem(hist_arr)
    ax.axvspan(th0, th1, facecolor="y", alpha=0.3)
    fig.canvas.draw()
    hist = np.array(fig.canvas.renderer._renderer)
    idxs = (image>=th0) & (image<=th1)
    pseudo = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    pseudo[idxs, :] = (0, 255, 255)
    return pseudo, hist

###############################################################################
#              Cargar imagen, aplicar filtros y definir ventanas              #
###############################################################################
IMAGE_DIR = "../imgs/"
IMAGE_FILE = "rio.jpg"
original = cv2.imread(f"{IMAGE_DIR}{IMAGE_FILE}", cv2.IMREAD_GRAYSCALE)
pseudo = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
ISHX, ISHY = original.shape[1], original.shape[0]
hist_arr = hi.get_histogram(original)

fig, ax = plt.subplots(1,1)
ax.stem(hi.get_histogram(original)) 
ax.axvspan(0, 255, facecolor="y", alpha=0.3)
ax.set_title("Histograma")
fig.canvas.draw()
hist = np.array(fig.canvas.renderer._renderer)

WINDOW_NAME = "Procesamiento en pseudo-color"
cv2.namedWindow(WINDOW_NAME)
cvui.init(WINDOW_NAME)

frame = np.zeros((ISHY+4*PADY+SLIDER_HEIGHT, 
                  ISHX+3*PADX+hist.shape[1], 3), np.uint8)

lth,uth = [0], [255]

###############################################################################
#                                 Bucle while                                 #
###############################################################################
while (True):
    if cv2.waitKey(1) == ord("q"):
        break

    cvui.context(WINDOW_NAME)
    frame[:]=(49, 52, 49)
    cvui.text(frame, PADX, 3*PADY+ISHY, "low", 0.5)
    cvui.trackbar(frame, 4*PADX, 2*PADY+ISHY, SLIDER_WIDTH, lth, 0, 255, 1, 
                  "%.0Lf", cvui.TRACKBAR_DISCRETE, 1)
    cvui.text(frame, PADX+ISHX, 3*PADY+ISHY, "high", 0.5)
    cvui.trackbar(frame, 5*PADX+ISHX, 2*PADY+ISHY, SLIDER_WIDTH, uth, 0, 255, 
                  1, "%.0Lf", cvui.TRACKBAR_DISCRETE, 1)

    pseudo, hist = apply_threshold(original, lth[0], uth[0], fig, ax)
    cvui.image(frame, PADX, PADY, pseudo)
    cvui.image(frame, 2*PADX+ISHX, PADY, 
               cv2.cvtColor(hist, cv2.COLOR_RGBA2BGR))

    cvui.update(WINDOW_NAME)
    cvui.imshow(WINDOW_NAME, frame)

