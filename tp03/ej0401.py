import cv2
import numpy as np
import sys
sys.path.append("../")
from cvui import cvui
from utils import filters as ut
from utils import operations as op


###############################################################################
#                      Constantes para ubicar elementos                       #
###############################################################################
PADX, PADY = 10, 10

###############################################################################
#              Cargar imagen, aplicar filtros y definir ventanas              #
###############################################################################
IMAGE_DIR = "../imgs/"
IMAGE_FILE = "esqueleto.tif"
imagen = cv2.imread(f"{IMAGE_DIR}{IMAGE_FILE}", cv2.IMREAD_GRAYSCALE)
original = np.copy(imagen)
original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
ISHX, ISHY = imagen.shape[1], imagen.shape[0]

WINDOW_NAME_1 = "Imagenes"
cv2.namedWindow(WINDOW_NAME_1)
cvui.init(WINDOW_NAME_1)

WINDOW_NAME_2 = "Panel de control para craneo"
cv2.namedWindow(WINDOW_NAME_2)
cvui.watch(WINDOW_NAME_2)

WINDOW_NAME_3 = "Panel de control para torso"
cv2.namedWindow(WINDOW_NAME_3)
cvui.watch(WINDOW_NAME_3)

frame1 = np.zeros((4*PADY+ISHY, 3*PADX+2*ISHX, 3), np.uint8)
frame2 = np.zeros((450, 350, 3), np.uint8)
frame3 = np.zeros((450, 350, 3), np.uint8)

tipo_str1, tipo_str2 = "CAJA", "CAJA"
kcaja1,    kcaja2    = [3],    [3]
kcruz1,    kcruz2    = [3],    [3]
kgauss1,   kgauss2   = [3],    [3]
sgauss1,   sgauss2   = [1.],   [1.]
kmedian1,  kmedian2  = [3],    [3]
hb_coef1,  hb_coef2  = [3],    [3]
hb_chk1,   hb_chk2   = [True], [True]
lim_inf1,  lim_inf2  = [ISHY//4], [ISHY//2]

###############################################################################
#                                 Bucle while                                 #
###############################################################################
while (True):
    if cv2.waitKey(1) == ord("q"):
        break

    cvui.context(WINDOW_NAME_2)
    frame2[:]=(49, 52, 49)
    cvui.text(frame2, PADX, PADY, f"Tipo actual: {tipo_str1}", 0.5)
    cvui.text(frame2, PADX, 5*PADY, f"Kernel para caja: ")
    if(cvui.trackbar(frame2, 13*PADX, 3*PADY, 150, kcaja1, 1, 11, 1, "%.0Lf",
                     cvui.TRACKBAR_DISCRETE, 2)):
        tipo_str1 = "CAJA"
    cvui.text(frame2, PADX, 10*PADY, f"Kernel para cruz: ")
    if(cvui.trackbar(frame2, 13*PADX, 8*PADY, 150, kcruz1, 1, 11, 1, "%.0Lf",
                     cvui.TRACKBAR_DISCRETE, 2)):
        tipo_str1 = "CRUZ"
    cvui.text(frame2, PADX, 15*PADY, f"Kernel para gaussiana: ")
    if(cvui.trackbar(frame2, 15*PADX, 13*PADY, 150, kgauss1, 1, 11, 1,
                     "%.0Lf", cvui.TRACKBAR_DISCRETE, 2)):
        tipo_str1 = "GAUSSIANA"
    cvui.text(frame2, PADX, 20*PADY, f"Sigma para gaussiana: ")
    if(cvui.trackbar(frame2, 15*PADX, 18*PADY, 175, sgauss1, 0, 5, 1,
                     "%.2Lf")):
        tipo_str1 = "GAUSSIANA"
    cvui.text(frame2, PADX, 25*PADY, f"Kernel para mediana: ")
    if(cvui.trackbar(frame2, 15*PADX, 23*PADY, 150, kmedian1, 1, 11, 1,
                     "%.0Lf", cvui.TRACKBAR_DISCRETE, 2)):
        tipo_str1 = "MEDIANA"
    cvui.text(frame2, PADX, 30*PADY, f"Coeficiente para HB: ")
    cvui.trackbar(frame2, 15*PADX, 28*PADY, 150, hb_coef1, 1, 5, 1, "%.2Lf")
    cvui.text(frame2, PADX, 35*PADY, f"Lim. Inf. craneo: ")
    cvui.trackbar(frame2, 15*PADX, 33*PADY, 150, lim_inf1, 1, ISHY, 1,
                  "%.0Lf", cvui.TRACKBAR_DISCRETE)
    cvui.checkbox(frame2, PADX, 40*PADY, "Alta potencia", hb_chk1);
    cvui.update(WINDOW_NAME_2)
    cvui.imshow(WINDOW_NAME_2, frame2)

    cvui.context(WINDOW_NAME_3)
    frame3[:]=(49, 52, 49)
    cvui.text(frame3, PADX, PADY, f"Tipo actual: {tipo_str2}", 0.5)
    cvui.text(frame3, PADX, 5*PADY, f"Kernel para caja: ")
    if(cvui.trackbar(frame3, 13*PADX, 3*PADY, 150, kcaja2, 1, 11, 1, "%.0Lf",
                     cvui.TRACKBAR_DISCRETE, 2)):
        tipo_str2 = "CAJA"
    cvui.text(frame3, PADX, 10*PADY, f"Kernel para cruz: ")
    if(cvui.trackbar(frame3, 13*PADX, 8*PADY, 150, kcruz2, 1, 11, 1, "%.0Lf",
                     cvui.TRACKBAR_DISCRETE, 2)):
        tipo_str2 = "CRUZ"
    cvui.text(frame3, PADX, 15*PADY, f"Kernel para gaussiana: ")
    if(cvui.trackbar(frame3, 15*PADX, 13*PADY, 150, kgauss2, 1, 11, 1,
                     "%.0Lf", cvui.TRACKBAR_DISCRETE, 2)):
        tipo_str2 = "GAUSSIANA"
    cvui.text(frame3, PADX, 20*PADY, f"Sigma para gaussiana: ")
    if(cvui.trackbar(frame3, 15*PADX, 18*PADY, 175, sgauss2, 0, 5, 1,
                     "%.2Lf")):
        tipo_str2 = "GAUSSIANA"
    cvui.text(frame3, PADX, 25*PADY, f"Kernel para mediana: ")
    if(cvui.trackbar(frame3, 15*PADX, 23*PADY, 150, kmedian2, 1, 11, 1,
                     "%.0Lf", cvui.TRACKBAR_DISCRETE, 2)):
        tipo_str2 = "MEDIANA"
    cvui.text(frame3, PADX, 30*PADY, f"Coeficiente para HB: ")
    cvui.trackbar(frame3, 15*PADX, 28*PADY, 150, hb_coef2, 1, 5, 1, "%.2Lf")
    cvui.text(frame3, PADX, 35*PADY, f"Lim. Inf. torso: ")
    cvui.trackbar(frame3, 15*PADX, 33*PADY, 150, lim_inf2, 1, ISHY, 1,
                  "%.0Lf", cvui.TRACKBAR_DISCRETE)
    cvui.checkbox(frame3, PADX, 40*PADY, "Alta potencia", hb_chk2);
    cvui.update(WINDOW_NAME_3)
    cvui.imshow(WINDOW_NAME_3, frame3)

    cvui.context(WINDOW_NAME_1)
    frame1[:]=(49, 52, 49)
    cvui.text(frame1, PADX, PADY, f"Imagen original")
    cvui.image(frame1, PADX, 3*PADY, original)
    cvui.text(frame1, 2*PADX+ISHX, PADY, f"Imagen realzada")
    if tipo_str1=="CAJA":
        filtrada1 = cv2.filter2D(imagen[:lim_inf1[0],:], -1,
                                 ut.generateKernel(ut.KBOXLOW, (kcaja1[0], 
                                                                kcaja1[0])))
    elif tipo_str1=="CRUZ":
        filtrada1 = cv2.filter2D(imagen[:lim_inf1[0],:], -1,
                                 ut.generateKernel(ut.KCROSSLOW, (kcruz1[0], 
                                                                  kcruz1[0])))
    elif tipo_str1=="GAUSSIANA":
        filtrada1 = cv2.GaussianBlur(imagen[:lim_inf1[0],:], (kgauss1[0],
                                                              kgauss1[0]),
                                    sgauss1[0])
    elif tipo_str1=="MEDIANA":
        filtrada1 = cv2.medianBlur(imagen[:lim_inf1[0],:], kmedian1[0])

    if tipo_str2=="CAJA":
        filtrada2 = cv2.filter2D(imagen[lim_inf1[0]:lim_inf2[0],:], -1,
                                 ut.generateKernel(ut.KBOXLOW, (kcaja2[0], 
                                                                kcaja2[0])))
    elif tipo_str2=="CRUZ":
        filtrada2 = cv2.filter2D(imagen[lim_inf1[0]:lim_inf2[0],:], -1,
                                 ut.generateKernel(ut.KCROSSLOW, (kcruz2[0], 
                                                                  kcruz2[0])))
    elif tipo_str2=="GAUSSIANA":
        filtrada2 = cv2.GaussianBlur(imagen[lim_inf1[0]:lim_inf2[0],:],
                                     (kgauss2[0], kgauss2[0]), sgauss2[0])
    elif tipo_str2=="MEDIANA":
        filtrada2 = cv2.medianBlur(imagen[lim_inf1[0]:lim_inf2[0],:],
                                   kmedian2[0])
    coef1 = hb_coef1[0] if hb_chk1[0] else 1.0
    coef2 = hb_coef2[0] if hb_chk2[0] else 1.0
    realzada = np.zeros_like(imagen)
    realzada[:lim_inf1[0],:] = op.diferencia(coef1*imagen[:lim_inf1[0],:
                                                          ].astype("float"),
                                             filtrada1.astype("float"), "no")
    realzada[lim_inf1[0]:lim_inf2[0],:
             ] = op.diferencia(coef2*imagen[lim_inf1[0]:lim_inf2[0],
                                            :].astype("float"),
                               filtrada2.astype("float"), "no")
    realzada[lim_inf2[0]:, :]=imagen[lim_inf2[0]:,:]
    cvui.image(frame1, 2*PADX+ISHX, 3*PADY,
               cv2.cvtColor(realzada.astype("uint8"), cv2.COLOR_GRAY2BGR))
    cvui.update(WINDOW_NAME_1)
    cvui.imshow(WINDOW_NAME_1, frame1)
    #cvui.imshow(WINDOW_NAME_1, cv2.cvtColor(realzada.astype("uint8"),
    #                                        cv2.COLOR_GRAY2BGR))
