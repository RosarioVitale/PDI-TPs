import cv2
import numpy as np
import sys
sys.path.append("../")
from cvui import cvui
from utils import filters as ut


###############################################################################
#                      Constantes para ubicar elementos                       #
###############################################################################
PADX, PADY = 10, 10

###############################################################################
#              Cargar imagen, aplicar filtros y definir ventanas              #
###############################################################################
IMAGE_DIR = "../imgs/"
IMAGE_FILE = "clown.jpg"
imagen = cv2.imread(f"{IMAGE_DIR}{IMAGE_FILE}", cv2.IMREAD_GRAYSCALE)
original = np.copy(imagen)
original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
ISHX, ISHY = imagen.shape[1], imagen.shape[0]

WINDOW_NAME_1 = "Imagenes"
cv2.namedWindow(WINDOW_NAME_1)
cvui.init(WINDOW_NAME_1)

WINDOW_NAME_2 = "Panel de control"
cv2.namedWindow(WINDOW_NAME_2)
cvui.watch(WINDOW_NAME_2)

frame1 = np.zeros((4*PADY+ISHY, 3*PADX+2*ISHX, 3), np.uint8)
frame2 = np.zeros((150, 300, 3), np.uint8)

tipo_str = "CAJA"
kcaja = [3]
kcruz = [3]

###############################################################################
#                                 Bucle while                                 #
###############################################################################
while (True):
    if cv2.waitKey(1) == ord("q"):
        break

    cvui.context(WINDOW_NAME_2)
    frame2[:]=(49, 52, 49)
    cvui.text(frame2, PADX, PADY, f"Tipo actual: {tipo_str}", 0.5)

    cvui.text(frame2, PADX, 5*PADY, f"Kernel para caja: ")
    if(cvui.trackbar(frame2, 13*PADX, 3*PADY, 150, kcaja, 1, 11, 1, "%.0Lf",
                     cvui.TRACKBAR_DISCRETE, 2)):
        tipo_str = "CAJA"

    cvui.text(frame2, PADX, 10*PADY, f"Kernel para cruz: ")
    if(cvui.trackbar(frame2, 13*PADX, 8*PADY, 150, kcruz, 1, 11, 1, "%.0Lf",
                     cvui.TRACKBAR_DISCRETE, 2)):
        tipo_str = "CRUZ"

    cvui.update(WINDOW_NAME_2)
    cvui.imshow(WINDOW_NAME_2, frame2)

    cvui.context(WINDOW_NAME_1)
    frame1[:]=(49, 52, 49)
    cvui.text(frame1, PADX, PADY, f"Imagen original")
    cvui.image(frame1, PADX, 3*PADY, original)
    cvui.text(frame1, 2*PADX+ISHX, PADY, f"Imagen filtrada")
    if tipo_str=="CAJA":
        filtrada = cv2.filter2D(imagen, -1, 
                                ut.generateKernel(ut.KBOXLOW, (kcaja[0],
                                                               kcaja[0])))
    elif tipo_str=="CRUZ":
        filtrada = cv2.filter2D(imagen, -1, 
                                ut.generateKernel(ut.KCROSSLOW, (kcruz[0], 
                                                                 kcruz[0])))
    cvui.image(frame1, 2*PADX+ISHX, 3*PADY, cv2.cvtColor(filtrada,
                                                         cv2.COLOR_GRAY2BGR))
    cvui.update(WINDOW_NAME_1)
    cvui.imshow(WINDOW_NAME_1, frame1)
