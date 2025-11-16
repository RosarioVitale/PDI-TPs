import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from cvui import cvui
import utils.noise as ns
import utils.histogram as ht

PADX, PADY, ISHX, ISHY, SSHX = 10, 10, 600, 600, 200

imagen = np.zeros((ISHY, ISHX), dtype="uint8")
#imagen[:,:SSHX]=180
#imagen[:,SSHX:2*SSHX]=120
#imagen[:,2*SSHX:]=60
imagen[:ISHX//2,:ISHY//2]=210
imagen[ISHX//2:,:ISHY//2]=100
imagen[:ISHX//2,ISHY//2:]=10
imagen[ISHX//2:,ISHY//2:]=150

noised = np.copy(imagen)

fig = plt.figure(figsize=(5,5))
fig.suptitle("Histograma")
ax = plt.subplot(1,1,1)
hist = ht.get_histogram(imagen)
hplt = ax.plot(hist)
fig.canvas.draw()
mplot = np.array(fig.canvas.renderer.buffer_rgba())

frame = np.zeros((400, 450, 3), np.uint8)

tipo_str = "IMPULSIVO"
imp_pa = [0.1]
imp_pb = [0.1]
gau_mn = [0]
gau_st = [1]
uni_a  = [108]
uni_b  = [148]
exp_a  = [0.05]

WINDOW_NAME_1 = "Imagen"
cv2.namedWindow(WINDOW_NAME_1)
cvui.init(WINDOW_NAME_1)
WINDOW_NAME_2 = "Histograma"
cv2.namedWindow(WINDOW_NAME_2)
cvui.watch(WINDOW_NAME_2)
WINDOW_NAME_3 = "Panel de control"
cv2.namedWindow(WINDOW_NAME_3)
cvui.watch(WINDOW_NAME_3)

calcular = True
while (True):
    if cv2.waitKey(1) == ord("q"):
        break
    cvui.context(WINDOW_NAME_3)
    frame[:]=(49, 52, 49)
    cvui.text(frame, PADX, PADY, f"Tipo actual: {tipo_str}", 0.5)
    cvui.text(frame, PADX, 5*PADY, f"Pa (impulsivo): ")
    if(cvui.trackbar(frame, 15*PADX, 3*PADY, 250, imp_pa, 0, 1, 1, "%.2Lf")):
        tipo_str = "IMPULSIVO"
        calcular = True
    cvui.text(frame, PADX, 10*PADY, f"Pb (impulsivo): ")
    if(cvui.trackbar(frame, 15*PADX, 8*PADY, 250, imp_pb, 0, 1, 1, "%.2Lf")):
        tipo_str = "IMPULSIVO"
        calcular = True
    cvui.text(frame, PADX, 15*PADY, f"Media (gaussiano): ")
    if(cvui.trackbar(frame, 15*PADX, 13*PADY, 250, gau_mn, 0, 255, 1, "%.0Lf",
                     cvui.TRACKBAR_DISCRETE, 1)):
        tipo_str = "GAUSSIANO"
        calcular = True
    cvui.text(frame, PADX, 20*PADY, f"Std (gaussiano): ")
    if(cvui.trackbar(frame, 15*PADX, 18*PADY, 250, gau_st, 0, 255, 1, "%.0Lf",
                     cvui.TRACKBAR_DISCRETE, 1)):
        tipo_str = "GAUSSIANO"
        calcular = True
    cvui.text(frame, PADX, 25*PADY, f"Min (uniforme): ")
    if(cvui.trackbar(frame, 15*PADX, 23*PADY, 250, uni_a, 0, 255, 1, "%.0Lf",
                     cvui.TRACKBAR_DISCRETE, 1)):
        tipo_str = "UNIFORME"
        calcular = True
    cvui.text(frame, PADX, 30*PADY, f"Max (uniforme): ")
    if(cvui.trackbar(frame, 15*PADX, 28*PADY, 250, uni_b, 0, 255, 1, "%.0Lf",
                     cvui.TRACKBAR_DISCRETE, 1)):
        tipo_str = "UNIFORME"
        calcular = True
    cvui.text(frame, PADX, 35*PADY, f"A (exponencial): ")
    if(cvui.trackbar(frame, 15*PADX, 33*PADY, 250, exp_a, 0.01, 0.1, 1, 
                     "%.3Lf")):
        tipo_str = "EXPONENCIAL"
        calcular = True
    cvui.update(WINDOW_NAME_3)
    cvui.imshow(WINDOW_NAME_3, frame)

    if calcular==True:
        calcular = False
        if tipo_str=="IMPULSIVO":
            noised = ns.add_impulsive_noise(imagen, imp_pa[0], imp_pb[0])
        elif tipo_str=="GAUSSIANO":
            noised = ns.add_gaussian_noise(imagen, gau_mn[0], gau_st[0])
        elif tipo_str=="UNIFORME":
            noised = ns.add_uniform_noise(imagen, uni_a[0], uni_b[0])
        elif tipo_str=="EXPONENCIAL":
            noised = ns.add_exponential_noise(imagen, exp_a[0])

    cvui.context(WINDOW_NAME_2)
    hist = ht.get_histogram(noised)
    hplt[0].set_ydata(hist)
    ax.set(ylim=(min(hist),max(hist)))
    fig.canvas.draw()
    mplot = np.array(fig.canvas.renderer.buffer_rgba())
    cvui.update(WINDOW_NAME_2)
    cvui.imshow(WINDOW_NAME_2, cv2.cvtColor(mplot, cv2.COLOR_RGBA2BGR))
    cvui.context(WINDOW_NAME_1)
    cvui.update(WINDOW_NAME_1)
    cvui.imshow(WINDOW_NAME_1, noised)

