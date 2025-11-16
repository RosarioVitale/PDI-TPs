import cv2
import numpy as np
import sys
sys.path.append("../")
from cvui import cvui
from utils import fourier as uf
from utils import filters as ut
from utils.operations import diferencia


###############################################################################
#                      Constantes para ubicar elementos                       #
###############################################################################
PADX, PADY = 10, 10

###############################################################################
#              Cargar imagen, aplicar filtros y definir ventanas              #
###############################################################################
IMAGE_FILE = "../imgs/camaleon.tif"
imagen = cv2.imread(f"{IMAGE_FILE}", cv2.IMREAD_GRAYSCALE)
original = np.copy(imagen)
original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
ISHX, ISHY = imagen.shape[1], imagen.shape[0]

WINDOW_NAME_1 = "Imagenes"
cv2.namedWindow(WINDOW_NAME_1)
cvui.init(WINDOW_NAME_1)

WINDOW_NAME_2 = "Panel de control espacial"
cv2.namedWindow(WINDOW_NAME_2)
cvui.watch(WINDOW_NAME_2)

WINDOW_NAME_3 = "Panel de control frecuencial"
cv2.namedWindow(WINDOW_NAME_3)
cvui.watch(WINDOW_NAME_3)

frame1 = np.zeros((4*PADY+ISHY, 4*PADX+3*ISHX, 3), np.uint8)
frame2 = np.zeros((320, 350, 3), np.uint8)
frame3 = np.zeros((420, 350, 3), np.uint8)

tipo_str = "GAUSSIANA"
kcaja, kcruz, kgauss, sgauss = [3], [3], [3], [1.]
kmedian, hb_coef, hb_chk = [3], [2], [True]

tipo_frec, filt_enum = "GAUSSIANO", uf.GAUSSIAN_FILTER
A_hbf, R0_i, R0_B, n_B, s_G, hbf_chk = [500], [10], [10], [2], [100], [True]
ehf_B, ehf_chk = [1000.], [False]
params = {"sigma": s_G[0], "A":A_hbf[0], "B":ehf_B[0]}

###############################################################################
#                                 Bucle while                                 #
###############################################################################
while (True):
    if cv2.waitKey(1) == ord("q"):
        break

    cvui.context(WINDOW_NAME_2)
    frame2[:]=(49, 52, 49)
    cvui.text(frame2, PADX, PADY, f"Tipo actual: {tipo_str}", 0.5)
    cvui.text(frame2, PADX, 5*PADY, f"Kernel para gaussiana: ")
    if(cvui.trackbar(frame2, 15*PADX, 3*PADY, 150, kgauss, 1, 11, 1, "%.0Lf",
                     cvui.TRACKBAR_DISCRETE, 2)):
        tipo_str = "GAUSSIANA"
    cvui.text(frame2, PADX, 10*PADY, f"Sigma para gaussiana: ")
    if(cvui.trackbar(frame2, 15*PADX, 8*PADY, 175, sgauss, 0, 5, 1, "%.2Lf")):
        tipo_str = "GAUSSIANA"
    cvui.text(frame2, PADX, 15*PADY, f"Kernel para mediana: ")
    if(cvui.trackbar(frame2, 15*PADX, 13*PADY, 150, kmedian, 1, 11, 1, "%.0Lf",
                     cvui.TRACKBAR_DISCRETE, 2)):
        tipo_str = "MEDIANA"
    cvui.text(frame2, PADX, 20*PADY, f"Coeficiente para HB: ")
    cvui.trackbar(frame2, 15*PADX, 18*PADY, 150, hb_coef, 0, 5, 1, "%.2Lf")
    cvui.checkbox(frame2, PADX, 25*PADY, "Alta potencia", hb_chk);
    cvui.update(WINDOW_NAME_2)
    cvui.imshow(WINDOW_NAME_2, frame2)

    cvui.context(WINDOW_NAME_3)
    frame3[:]=(49, 52, 49)
    cvui.text(frame3, PADX, PADY, f"Tipo actual: {tipo_frec}", 0.5)
    cvui.text(frame3, PADX, 5*PADY, f"R0 para ideal: ")
    if(cvui.trackbar(frame3, 13*PADX, 3*PADY, 175, R0_i, 0,ISHX/2,1,"%.2Lf")):
        tipo_frec, filt_enum = "IDEAL", uf.IDEAL_FILTER
        params["R0"] = R0_i[0]
    cvui.text(frame3, PADX, 10*PADY, f"R0 para Butterworth: ")
    if(cvui.trackbar(frame3, 13*PADX, 8*PADY, 175, R0_B, 0,ISHX/2,1,"%.2Lf")):
        tipo_frec, filt_enum = "BUTTERWORTH", uf.BUTTERWORTH_FILTER
        params["R0"] = R0_B[0]
        params["n"] = n_B[0]
    cvui.text(frame3, PADX, 15*PADY, f"N para Butterworth: ")
    if(cvui.trackbar(frame3, 13*PADX, 13*PADY, 175, n_B, 0,10,1,"%.2Lf")):
        tipo_frec, filt_enum = "BUTTERWORTH", uf.BUTTERWORTH_FILTER
        params["R0"] = R0_B[0]
        params["n"] = n_B[0]
    cvui.text(frame3, PADX, 20*PADY, f"S para Gaussiano: ")
    if(cvui.trackbar(frame3, 13*PADX, 18*PADY, 175, s_G, 1,ISHX/2,1,"%.2Lf")):
        tipo_frec, filt_enum = "GAUSSIANO", uf.GAUSSIAN_FILTER
        params["sigma"] = s_G[0]
    cvui.text(frame3, PADX, 25*PADY, f"Coeficiente para HB: ")
    if(cvui.trackbar(frame3, 13*PADX, 23*PADY, 150, A_hbf, 1, 500, 1, "%.2Lf")):
        params["A"] = A_hbf[0]
    cvui.text(frame3, PADX, 30*PADY, f"Coeficiente para EHF: ")
    if(cvui.trackbar(frame3, 13*PADX, 28*PADY, 150, ehf_B, 500, 1500, 1, "%.2Lf")):
        params["B"] = ehf_B[0]
    cvui.checkbox(frame3, PADX, 33*PADY, "Alta potencia", hbf_chk);
    cvui.checkbox(frame3, PADX, 38*PADY, "Enfasis alta frecuencia", ehf_chk);
    cvui.update(WINDOW_NAME_3)
    cvui.imshow(WINDOW_NAME_3, frame3)

    cvui.context(WINDOW_NAME_1)
    frame1[:]=(49, 52, 49)
    cvui.text(frame1, PADX, PADY, f"Imagen original")
    cvui.image(frame1, PADX, 3*PADY, original)
    cvui.text(frame1, 2*PADX+ISHX, PADY, f"Imagen filtrada (espacial)")
    if tipo_str=="CAJA":
        filtrada = cv2.filter2D(imagen, -1, ut.generateKernel(ut.KBOXLOW,
                                                              (kcaja[0],
                                                               kcaja[0])))
    elif tipo_str=="CRUZ":
        filtrada = cv2.filter2D(imagen, -1, ut.generateKernel(ut.KCROSSLOW,
                                                              (kcruz[0],
                                                               kcruz[0])))
    elif tipo_str=="GAUSSIANA":
        filtrada = cv2.GaussianBlur(imagen, (kgauss[0],kgauss[0]), sgauss[0])
    elif tipo_str=="MEDIANA":
        filtrada = cv2.medianBlur(imagen, kmedian[0])

    coef = hb_coef[0] if hb_chk[0] else 1.0
    realzada = diferencia(coef*imagen.astype("float"),
                          filtrada.astype("float"), "no")
    cvui.image(frame1, 2*PADX+ISHX, 3*PADY,
               cv2.cvtColor(realzada.astype("uint8"), cv2.COLOR_GRAY2BGR))
    cvui.text(frame1, 3*PADX+2*ISHX, PADY, f"Imagen filtrada (frecuencial)")
    fil_frec = np.zeros_like(imagen)
    params["B"]=ehf_B[0] if ehf_chk[0] else 1.0
    fil_frec = uf.apply_filter(imagen, filt_enum, params, False, hbf_chk[0])
    cvui.image(frame1, 3*PADX+2*ISHX, 3*PADY,
               cv2.cvtColor(fil_frec.astype("uint8"), cv2.COLOR_GRAY2BGR))
    cvui.update(WINDOW_NAME_1)
    cvui.imshow(WINDOW_NAME_1, frame1)
