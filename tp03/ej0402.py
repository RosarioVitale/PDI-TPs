import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from cvui import cvui
from utils import filters as ft
from utils import utils as ut


###############################################################################
#                      Constantes para ubicar elementos                       #
###############################################################################
PADX, PADY = 10, 10

###############################################################################
#                        Funciones para este ejercicio                        #
###############################################################################
def get_spl_data(X1, Y1, X2, Y2, imagen):
    """TODO: Docstring for get_spl_data.

    Parameters:
        (X1: TODO
        Y1: TODO
        X2: TODO
        Y2: TODO
    Returns:
        TODO

    """
    if X1==X2 and Y1==Y2:
        return [imagen[X1, Y1]]
    if X1==X2:
        return imagen[Y1:Y2, X1].tolist()
    if Y1==Y2:
        return imagen[Y1, X1:X2].tolist()
    Y = [int(X*(Y2-Y1)/(X2-X1)+Y1) for X in range(X2-X1)]
    X = [i for i in range(X1,X2)]
    return imagen[Y,X].tolist()

###############################################################################
#              Cargar imagen, aplicar filtros y definir ventanas              #
###############################################################################
IMAGE_DIR = "../imgs/"
IMAGE_FILE = "mariposa02.png"
imagen = cv2.imread(f"{IMAGE_DIR}{IMAGE_FILE}", cv2.IMREAD_GRAYSCALE)
original = np.copy(imagen)
original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
ISHX, ISHY = imagen.shape[1], imagen.shape[0]

WINDOW_NAME_1 = "Imagen original"
cv2.namedWindow(WINDOW_NAME_1)
cvui.init(WINDOW_NAME_1)

WINDOW_NAME_2 = "Panel de control"
cv2.namedWindow(WINDOW_NAME_2)
cvui.watch(WINDOW_NAME_2)

WINDOW_NAME_3 = "Pasa bajos"
cv2.namedWindow(WINDOW_NAME_3)
cvui.watch(WINDOW_NAME_3)

WINDOW_NAME_4 = "Bilateral"
cv2.namedWindow(WINDOW_NAME_4)
cvui.watch(WINDOW_NAME_4)

WINDOW_NAME_5 = "Seleccion de segmento"
cv2.namedWindow(WINDOW_NAME_5)
cvui.watch(WINDOW_NAME_5)

WINDOW_NAME_6 = "Perfiles de intensidad"
cv2.namedWindow(WINDOW_NAME_6)
cvui.watch(WINDOW_NAME_6)

frame1 = np.zeros((450, 350, 3), np.uint8)
frame2 = np.zeros((250, 350, 3), np.uint8)

#####################################################
#  contenedores de variables para panel de control  #
#####################################################
tipo_str = "GAUSSIANA"
kcaja = [3]
kcruz = [3]
kgauss = [3]
sgauss = [1.]
kmedian = [3]
dbil = [3]
sbilC = [1.]
sbilS = [1.]

###########################################################
#  contenedores de variables para perfiles de intensidad  #
###########################################################
draw_seg = [False]
segX1 = [1]
segY1 = [1]
segX2 = [100]
segY2 = [100]

###############################################################
#  Configuraciones para el grafico de perfiles de intensidad  #
###############################################################
fig, ax = plt.subplots(1, 1)
seg = ut.get_segment(imagen, segX1[0], segY1[0], segX2[0], segY2[0])
ax.plot(seg, "b*-", label="Original")
ax.plot(seg, "go-", label="Filtrada")
ax.plot(seg, "r^-", label="Bilateral")
fig.canvas.draw()
profiles = np.array(fig.canvas.renderer._renderer)

###############################################################################
#                                 Bucle while                                 #
###############################################################################
while (True):
    if cv2.waitKey(1) == ord("q"):
        break

    cvui.context(WINDOW_NAME_2)
    frame1[:]=(49, 52, 49)
    cvui.text(frame1, PADX, PADY, f"Tipo actual: {tipo_str}", 0.5)
    cvui.text(frame1, PADX, 5*PADY, f"Kernel para caja: ")
    if(cvui.trackbar(frame1, 13*PADX, 3*PADY, 150, kcaja, 1, 11, 1, "%.0Lf",
                     cvui.TRACKBAR_DISCRETE, 2)):
        tipo_str = "CAJA"
    cvui.text(frame1, PADX, 10*PADY, f"Kernel para cruz: ")
    if(cvui.trackbar(frame1, 13*PADX, 8*PADY, 150, kcruz, 1, 11, 1, "%.0Lf",
                     cvui.TRACKBAR_DISCRETE, 2)):
        tipo_str = "CRUZ"
    cvui.text(frame1, PADX, 15*PADY, f"Kernel para gaussiana: ")
    if(cvui.trackbar(frame1, 15*PADX, 13*PADY, 150, kgauss, 1, 11, 1, "%.0Lf",
                     cvui.TRACKBAR_DISCRETE, 2)):
        tipo_str = "GAUSSIANA"
    cvui.text(frame1, PADX, 20*PADY, f"Sigma para gaussiana: ")
    if(cvui.trackbar(frame1, 15*PADX, 18*PADY, 175, sgauss, 0, 250, 1,
                     "%.2Lf")):
        tipo_str = "GAUSSIANA"
    cvui.text(frame1, PADX, 25*PADY, f"Kernel para mediana: ")
    if(cvui.trackbar(frame1, 15*PADX, 23*PADY, 150, kmedian, 1, 11, 1, "%.0Lf",
                     cvui.TRACKBAR_DISCRETE, 2)):
        tipo_str = "MEDIANA"
    cvui.text(frame1, PADX, 30*PADY, f"D para Bilateral: ")
    cvui.trackbar(frame1, 15*PADX, 28*PADY, 150, dbil, 1, 11, 1, "%.0Lf",
                     cvui.TRACKBAR_DISCRETE, 2)
    cvui.text(frame1, PADX, 35*PADY, f"SigmaC para Bilateral: ")
    cvui.trackbar(frame1, 15*PADX, 33*PADY, 175, sbilC, 0, 250, 1, "%.2Lf")
    cvui.text(frame1, PADX, 40*PADY, f"SigmaS para Bilateral: ")
    cvui.trackbar(frame1, 15*PADX, 38*PADY, 175, sbilS, 0, 250, 1, "%.2Lf")
    cvui.update(WINDOW_NAME_2)
    cvui.imshow(WINDOW_NAME_2, frame1)

    if tipo_str=="CAJA":
        filtrada = cv2.filter2D(imagen, -1, ft.generateKernel(ft.KBOXLOW,
                                                              (kcaja[0],
                                                               kcaja[0])))
    elif tipo_str=="CRUZ":
        filtrada = cv2.filter2D(imagen, -1, ft.generateKernel(ft.KCROSSLOW,
                                                              (kcruz[0],
                                                               kcruz[0])))
    elif tipo_str=="GAUSSIANA":
        filtrada = cv2.GaussianBlur(imagen, (kgauss[0],kgauss[0]), sgauss[0])
    elif tipo_str=="MEDIANA":
        filtrada = cv2.medianBlur(imagen, kmedian[0])
    bilateral = cv2.bilateralFilter(imagen, dbil[0], sbilC[0], sbilS[0])
    original=np.copy(imagen)

    cvui.context(WINDOW_NAME_5)
    frame2[:]=(49, 52, 49)
    cvui.checkbox(frame2, PADX, PADY, "Dibujar segmento:", draw_seg)
    cvui.text(frame2, PADX, 5*PADY, f"X1:")
    cvui.trackbar(frame2, 2*PADX, 3*PADY, 230, segX1, 0, ISHX, 1, "%.0Lf",
                  cvui.TRACKBAR_DISCRETE, 1)
    cvui.counter(frame2, 25*PADX, 4*PADY, segX1)
    cvui.text(frame2, PADX, 10*PADY, f"Y1:")
    cvui.trackbar(frame2, 2*PADX, 8*PADY, 230, segY1, 0, ISHY, 1, "%.0Lf",
                  cvui.TRACKBAR_DISCRETE, 1)
    cvui.counter(frame2, 25*PADX, 9*PADY, segY1)
    cvui.text(frame2, PADX, 15*PADY, f"X2:")
    cvui.trackbar(frame2, 2*PADX, 13*PADY, 230, segX2, 0, ISHX, 1, "%.0Lf",
                  cvui.TRACKBAR_DISCRETE, 1)
    cvui.counter(frame2, 25*PADX, 14*PADY, segX2)
    cvui.text(frame2, PADX, 20*PADY, f"Y2:")
    cvui.trackbar(frame2, 2*PADX, 18*PADY, 230, segY2, 0, ISHY, 1, "%.0Lf",
                  cvui.TRACKBAR_DISCRETE, 1)
    cvui.counter(frame2, 25*PADX, 19*PADY, segY2)

    if segX1[0]<segX2[0]:
        X1, X2=segX1[0], segX2[0]
    else:
        X1, X2=segX2[0], segX1[0]
    if segY1[0]<segY2[0]:
        Y1, Y2=segY1[0], segY2[0]
    else:
        Y1, Y2=segY2[0], segY1[0]

    sp_pts1=np.array(ut.get_segment(imagen, X1, Y1, X2, Y2))
    sp_pts2=np.array(ut.get_segment(filtrada, X1, Y1, X2, Y2))
    sp_pts3=np.array(ut.get_segment(bilateral, X1, Y1, X2, Y2))
    ax.clear()
    ax.plot(sp_pts1, "b*-", label="Original")
    ax.plot(sp_pts2, "go-", label="Filtrada")
    ax.plot(sp_pts3, "r^-", label="Bilateral")
    ax.legend()
    fig.canvas.draw()
    profiles = np.array(fig.canvas.renderer._renderer)
    cvui.update(WINDOW_NAME_5)
    cvui.imshow(WINDOW_NAME_5, frame2)

    if draw_seg[0]:
        original=cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        cv2.line(original, (segX1[0], segY1[0]), (segX2[0], segY2[0]),
                 (255,0,0), 2)
        filtrada=cv2.cvtColor(filtrada, cv2.COLOR_GRAY2BGR)
        cv2.line(filtrada, (segX1[0], segY1[0]), (segX2[0], segY2[0]),
                 (0,255,0), 2)
        bilateral=cv2.cvtColor(bilateral, cv2.COLOR_GRAY2BGR)
        cv2.line(bilateral, (segX1[0], segY1[0]), (segX2[0], segY2[0]),
                 (0,0,255), 2)

    cvui.update(WINDOW_NAME_1)
    cvui.imshow(WINDOW_NAME_1, original)

    cvui.update(WINDOW_NAME_3)
    cvui.imshow(WINDOW_NAME_3, filtrada)

    cvui.update(WINDOW_NAME_4)
    cvui.imshow(WINDOW_NAME_4, bilateral)

    cvui.update(WINDOW_NAME_6)
    cvui.imshow(WINDOW_NAME_6, cv2.cvtColor(profiles, cv2.COLOR_BGR2RGB))

