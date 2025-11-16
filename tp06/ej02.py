import cv2
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("..")
from cvui import cvui
import utils.noise as ns
import utils.histogram as ht
from utils.utils import error_mse as mse

PADX, PADY = 10, 10

IMAGE_DIR = "../imgs/"
IMAGE_FILE = "sangre.jpg"
imagen = cv2.imread(f"{IMAGE_DIR}{IMAGE_FILE}", cv2.IMREAD_GRAYSCALE)
noised = np.copy(imagen)
ISHY, ISHX = imagen.shape
fig = plt.figure(figsize=(12,4))
fig.suptitle("Histogramas")
ax1 = plt.subplot(1,3,1)
ax1.set_title("Original")
hist1 = ht.get_histogram(imagen)
hplt1 = ax1.plot(hist1)
ax2 = plt.subplot(1,3,2)
ax2.set_title("Ruidosa")
hist2 = ht.get_histogram(imagen)
hplt2 = ax2.plot(hist2)
ax3 = plt.subplot(1,3,3)
ax3.set_title("Filtrada")
hist3 = ht.get_histogram(imagen)
hplt3 = ax3.plot(hist3)
fig.canvas.draw()
mplot = np.array(fig.canvas.renderer.buffer_rgba())

frame1 = np.zeros((300, 450, 3), np.uint8)
frame2 = np.zeros((300, 450, 3), np.uint8)

imp_chk= [True]
imp_pa = [0.01]
imp_pb = [0.01]
gau_chk= [True]
gau_mn = [0]
gau_st = [1]

geo_chk= [False]
geo_M  = [3]
cth_chk= [False]
cth_M  = [3]
cth_Q  = [2]
inv_chk= [True]

WINDOW_NAME_1 = "Imagen con ruido"
cv2.namedWindow(WINDOW_NAME_1)
cvui.init(WINDOW_NAME_1)
WINDOW_NAME_2 = "Histogramas"
cv2.namedWindow(WINDOW_NAME_2)
cvui.watch(WINDOW_NAME_2)
WINDOW_NAME_3 = "Panel de control del ruido"
cv2.namedWindow(WINDOW_NAME_3)
cvui.watch(WINDOW_NAME_3)
WINDOW_NAME_4 = "Panel de control del filtro"
cv2.namedWindow(WINDOW_NAME_4)
cvui.watch(WINDOW_NAME_4)
WINDOW_NAME_5 = "Imagen filtrada"
cv2.namedWindow(WINDOW_NAME_5)
cvui.watch(WINDOW_NAME_5)

calcular = True
filtrar = True
imp_chk_prev = imp_chk[0]
gau_chk_prev = gau_chk[0]
geo_chk_prev = geo_chk[0]
cth_chk_prev = cth_chk[0]
inv_chk_prev = inv_chk[0]
while (True):
    if cv2.waitKey(1) == ord("q"):
        break
    cvui.context(WINDOW_NAME_3)
    frame1[:]=(49, 52, 49)
    cvui.checkbox(frame1, PADX, PADY, "Aplicar impulsivo", imp_chk)
    cvui.text(frame1, PADX, 5*PADY, f"Pa (impulsivo): ")
    if(cvui.trackbar(frame1, 15*PADX, 3*PADY, 250, imp_pa, 0, 0.1, 1, "%.4Lf")):
        calcular = True
        filtrar = True
    cvui.text(frame1, PADX, 10*PADY, f"Pb (impulsivo): ")
    if(cvui.trackbar(frame1, 15*PADX, 8*PADY, 250, imp_pb, 0, 0.1, 1, "%.4Lf")):
        calcular = True
        filtrar = True
    cvui.checkbox(frame1, PADX, 15*PADY, "Aplicar gaussiano", gau_chk)
    cvui.text(frame1, PADX, 20*PADY, f"Media (gaussiano): ")
    if(cvui.trackbar(frame1, 15*PADX, 18*PADY, 250, gau_mn, 0, 10, 1, "%.0Lf",
                     cvui.TRACKBAR_DISCRETE, 1)):
        calcular = True
        filtrar = True
    cvui.text(frame1, PADX, 25*PADY, f"Std (gaussiano): ")
    if(cvui.trackbar(frame1, 15*PADX, 23*PADY, 250, gau_st, 0, 50, 1, "%.0Lf",
                     cvui.TRACKBAR_DISCRETE, 1)):
        calcular = True
        filtrar = True
    cvui.update(WINDOW_NAME_3)
    cvui.imshow(WINDOW_NAME_3, frame1)

    if imp_chk[0]!=imp_chk_prev:
        calcular = True
        filtrar = True
        imp_chk_prev=imp_chk[0]

    if gau_chk[0]!=gau_chk_prev:
        calcular = True
        filtrar = True
        gau_chk_prev=gau_chk[0]

    if calcular==True:
        calcular=False
        if gau_chk[0]:
            noised = ns.add_gaussian_noise(imagen, gau_mn[0], gau_st[0])
        else:
            noised = np.copy(imagen)
        if imp_chk[0]:
            noised = ns.add_impulsive_noise(noised, imp_pa[0], imp_pb[0])

    cvui.context(WINDOW_NAME_4)
    frame2[:]=(49, 52, 49)
    cvui.checkbox(frame2, PADX, PADY, "Aplicar media geometrica", geo_chk)
    cvui.text(frame2, PADX, 5*PADY, f"M (m. geom.): ")
    if(cvui.trackbar(frame2, 15*PADX, 3*PADY, 250, geo_M, 1, 20, 1, "%.0Lf",
                     cvui.TRACKBAR_DISCRETE, 1)):
        filtrar = True
    cvui.checkbox(frame2, PADX, 10*PADY, "Aplicar media contra-armonica",
                  cth_chk)
    cvui.text(frame2, PADX, 15*PADY, f"M (m. c. arm.): ")
    if(cvui.trackbar(frame2, 15*PADX, 13*PADY, 250, cth_M, 1, 20, 1, "%.0Lf",
                     cvui.TRACKBAR_DISCRETE, 1)):
        filtrar = True
    cvui.text(frame2, PADX, 20*PADY, f"Q (m. c. arm.): ")
    if(cvui.trackbar(frame2, 15*PADX, 18*PADY, 250, cth_Q, -10,10, 1, "%.2Lf")):
                     #cvui.TRACKBAR_DISCRETE, 1)):
        filtrar = True
    cvui.checkbox(frame2, PADX, 25*PADY, "Aplicar geometrica primero",
                  inv_chk)
    cvui.update(WINDOW_NAME_4)
    cvui.imshow(WINDOW_NAME_4, frame2)

    cvui.context(WINDOW_NAME_5)
    if geo_chk[0]!=geo_chk_prev:
        filtrar = True
        geo_chk_prev=geo_chk[0]

    if cth_chk[0]!=cth_chk_prev:
        filtrar = True
        cth_chk_prev=cth_chk[0]

    if inv_chk[0]!=inv_chk_prev:
        filtrar = True
        inv_chk_prev=inv_chk[0]

    if filtrar==True:
        filtered = np.copy(noised)
        filtrar=False
        if inv_chk[0]:
            if geo_chk[0]:
                filtered = ns.geometric_mean_filter(noised, geo_M[0])
            if cth_chk[0]:
                filtered = ns.contraharmonic_mean_filter(filtered, cth_M[0],
                                                         cth_Q[0])
        else:
            if cth_chk[0]:
                filtered = ns.contraharmonic_mean_filter(noised, cth_M[0],
                                                         cth_Q[0])
            if geo_chk[0]:
                filtered = ns.geometric_mean_filter(filtered, geo_M[0])
    cl_filt=cv2.cvtColor(filtered,cv2.COLOR_GRAY2BGR)
    mse_filt = mse(filtered, imagen)
    cv2.putText(cl_filt, f"ECM: {mse_filt:6.2f}", (3*PADX,3*PADY),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cvui.update(WINDOW_NAME_5)
    cvui.imshow(WINDOW_NAME_5, cl_filt)

    cvui.context(WINDOW_NAME_1)
    cvui.update(WINDOW_NAME_1)
    cl_nois=cv2.cvtColor(noised,cv2.COLOR_GRAY2BGR)
    mse_nois = mse(noised, imagen)
    cv2.putText(cl_nois, f"ECM: {mse_nois:6.2f}", (3*PADX,3*PADY),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cvui.imshow(WINDOW_NAME_1, cl_nois)

    cvui.context(WINDOW_NAME_2)
    hist2 = ht.get_histogram(noised)
    hplt2[0].set_ydata(hist2)
    ax2.set(ylim=(min(hist2)-10,max(hist2)+10))
    hist3 = ht.get_histogram(filtered)
    hplt3[0].set_ydata(hist3)
    ax3.set(ylim=(min(hist3)-10,max(hist3)+10))
    fig.canvas.draw()
    mplot = np.array(fig.canvas.renderer.buffer_rgba())
    cvui.update(WINDOW_NAME_2)
    cvui.imshow(WINDOW_NAME_2, cv2.cvtColor(mplot, cv2.COLOR_RGBA2BGR))

