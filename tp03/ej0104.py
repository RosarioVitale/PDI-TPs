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
IMAGE_FILE = "hubble.tif"

imagen = cv2.imread(f"{IMAGE_DIR}{IMAGE_FILE}", cv2.IMREAD_GRAYSCALE)
original = np.copy(imagen)
original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
ISHX, ISHY = imagen.shape[1], imagen.shape[0]

WINDOW_NAME_1 = "Hubble"
WINDOW_NAME_2 = "Panel de control"

cv2.namedWindow(WINDOW_NAME_1)
cv2.namedWindow(WINDOW_NAME_2)

cvui.init(WINDOW_NAME_1)
cvui.watch(WINDOW_NAME_2)

frame1 = np.zeros((4*PADY+ISHY, 4*PADX+3*ISHX, 3), np.uint8)
frame2 = np.zeros((350, 350, 3), np.uint8)

tipo_str = "CAJA"
kcaja = [3]
kcruz = [3]
kgauss = [3]
sgauss = [1.]
kmedian = [3]
umbral = [100]

while (True):
    if cv2.waitKey(1) == ord("q"):
        break

    #################################
    #  Ventana de panel de control  #
    #################################
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

    cvui.text(frame2, PADX, 15*PADY, f"Kernel para gaussiana: ")
    if(cvui.trackbar(frame2, 15*PADX, 13*PADY, 150, kgauss, 1, 11, 1, "%.0Lf",
                     cvui.TRACKBAR_DISCRETE, 2)):
        tipo_str = "GAUSSIANA"
    cvui.text(frame2, PADX, 20*PADY, f"Sigma para gaussiana: ")
    if(cvui.trackbar(frame2, 15*PADX, 18*PADY, 175, sgauss, 0, 5, 1, "%.2Lf")):
        tipo_str = "GAUSSIANA"

    cvui.text(frame2, PADX, 25*PADY, f"Kernel para mediana: ")
    if(cvui.trackbar(frame2, 15*PADX, 23*PADY, 150, kmedian, 1, 11, 1, "%.0Lf",
                     cvui.TRACKBAR_DISCRETE, 2)):
        tipo_str = "MEDIANA"

    cvui.text(frame2, PADX, 30*PADY, f"Umbral para mascara: ")
    cvui.trackbar(frame2, 15*PADX, 28*PADY, 200, umbral, 0, 255, 1, "%.0Lf")

    cvui.update(WINDOW_NAME_2)
    cvui.imshow(WINDOW_NAME_2, frame2)

    #########################
    #  Ventana de imagenes  #
    #########################
    cvui.context(WINDOW_NAME_1)
    frame1[:]=(49, 52, 49)
    cvui.text(frame1, PADX, PADY, f"Imagen original")
    cvui.image(frame1, PADX, 3*PADY, original)
    cvui.text(frame1, 2*PADX+ISHX, PADY, f"Imagen difuminada")
    if tipo_str=="CAJA":
        filtrada = cv2.filter2D(imagen, -1, 
                                ut.generateKernel(ut.KBOXLOW, (kcaja[0], 
                                                               kcaja[0])))
    elif tipo_str=="CRUZ":
        filtrada = cv2.filter2D(imagen, -1, 
                                ut.generateKernel(ut.KCROSSLOW, (kcruz[0], 
                                                                 kcruz[0])))
    elif tipo_str=="GAUSSIANA":
        filtrada = cv2.GaussianBlur(imagen, (kgauss[0],kgauss[0]), sgauss[0])
    elif tipo_str=="MEDIANA":
        filtrada = cv2.medianBlur(imagen, kmedian[0])
    cvui.image(frame1, 2*PADX+ISHX, 3*PADY, cv2.cvtColor(filtrada,
                                                         cv2.COLOR_GRAY2BGR))
    cvui.text(frame1, 3*PADX+2*ISHX, PADY, f"Objetos grandes")
    mascara = np.zeros_like(filtrada)
    mascara[filtrada>umbral[0]] = 1.0
    imgmask = (mascara*255).astype("uint8")
    cvui.image(frame1, 3*PADX+2*ISHX, 3*PADY,
               cv2.cvtColor(imgmask, cv2.COLOR_GRAY2BGR))
    cvui.update(WINDOW_NAME_1)
    cvui.imshow(WINDOW_NAME_1, frame1)
