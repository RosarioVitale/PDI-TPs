import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from cvui import cvui
from utils import utils as ut
from utils import histogram as hi
from utils import color as co

###############################################################################
#                      Constantes para ubicar elementos                       #
###############################################################################
PADX, PADY, SLIDER_WIDTH, SLIDER_HEIGHT, TEXT_HEIGHT = 10, 10, 255, 30, 30

###############################################################################
#                        Funciones para este ejercicio                        #
###############################################################################

###############################################################################
#              Abrir camara, aplicar filtros y definir ventanas               #
###############################################################################
video = cv2.VideoCapture(0)
imagen = video.read()[1]
img_bgr = np.copy(imagen)
img_hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
ISHX, ISHY = imagen.shape[1], imagen.shape[0]

WINDOW_NAME_1 = "Imagenes"
cv2.namedWindow(WINDOW_NAME_1)
cvui.init(WINDOW_NAME_1)

WINDOW_NAME_2 = "Panel de control"
cv2.namedWindow(WINDOW_NAME_2)
cvui.watch(WINDOW_NAME_2)

WINDOW_NAME_3 = "Histogramas"
cv2.namedWindow(WINDOW_NAME_3)
cvui.watch(WINDOW_NAME_3)

frame1 = np.zeros((ISHY+4*PADY+TEXT_HEIGHT, 3*ISHX+4*PADX, 3), np.uint8)
frame2 = np.zeros((7*SLIDER_HEIGHT+20*PADY+TEXT_HEIGHT,3*SLIDER_WIDTH+11*PADX, 
                   3), np.uint8)

#####################################################
#  contenedores de variables para panel de control  #
#####################################################
tipo_str = "RGB sphere" # "RGB sphere" o "RGB cube" o "HSV cube"
RGB_r, a_R, a_G, a_B = [0], [0], [0], [0]
R_min, R_max, G_min, G_max, B_min, B_max = [0], [255], [0], [255], [0], [255]
H_min, H_max, S_min, S_max, V_min, V_max = [0], [180], [0], [255], [0], [255]
img_seg, mask = co.color_slicing_sphere(img_bgr, [a_B[0], a_G[0], a_R[0]], 
                                        RGB_r[0])

###############################################################
#  Configuraciones para el grafico de perfiles de intensidad  #
###############################################################
fig, ax = plt.subplots(2, 3)
hist_bgr = hi.get_histogram(img_bgr, mode="BGR")
hist_hsv = hi.get_histogram(img_hsv, mode="HSV")
ax[0,0].stem(hist_bgr[2], markerfmt="", linefmt="r")
ax[0,0].set_title("Histograma R")
ax[0,0].axvspan(R_min[0], R_max[0], facecolor="r", alpha=0.3)
ax[0,1].stem(hist_bgr[1], markerfmt="", linefmt="g")
ax[0,1].set_title("Histograma G")
ax[0,1].axvspan(G_min[0], G_max[0], facecolor="g", alpha=0.3)
ax[0,2].stem(hist_bgr[0], markerfmt="", linefmt="b")
ax[0,2].set_title("Histograma B")
ax[0,2].axvspan(B_min[0], B_max[0], facecolor="b", alpha=0.3)
ax[1,0].stem(hist_hsv[0], markerfmt="", linefmt="darkorange")
ax[1,0].set_title("Histograma H")
ax[1,0].axvspan(H_min[0], H_max[0], facecolor="darkorange", alpha=0.3)
ax[1,1].stem(hist_hsv[1], markerfmt="", linefmt="limegreen")
ax[1,1].set_title("Histograma S")
ax[1,1].axvspan(S_min[0], S_max[0], facecolor="limegreen", alpha=0.3)
ax[1,2].stem(hist_hsv[2], markerfmt="", linefmt="royalblue")
ax[1,2].set_title("Histograma V")
ax[1,2].axvspan(V_min[0], V_max[0], facecolor="royalblue", alpha=0.3)
fig.canvas.draw()
histograms = np.array(fig.canvas.renderer._renderer)

###############################################################################
#                                 Bucle while                                 #
###############################################################################
while (True):
    if cv2.waitKey(1) == ord("q"):
        break

    cvui.context(WINDOW_NAME_2)
    frame2[:]=(49, 52, 49)
    cvui.text(frame2, PADX, PADY, f"Tipo actual: {tipo_str}", 0.5)

    # RGB slicing sphere
    cvui.text(frame2, PADX, 5*PADY, f"Rad: ")
    cvui.text(frame2, PADX, 7*PADY+SLIDER_HEIGHT, f"aR: ")
    cvui.text(frame2, PADX*5+SLIDER_WIDTH, 7*PADY+SLIDER_HEIGHT, f"aG: ")
    cvui.text(frame2, PADX*8+2*SLIDER_WIDTH, 7*PADY+SLIDER_HEIGHT, f"aB: ")
    if(cvui.trackbar(frame2, 3*PADX, 3*PADY, 500, RGB_r, 0, 500, 1, "%.1Lf") or
       cvui.trackbar(frame2, 3*PADX, 5*PADY+SLIDER_HEIGHT, SLIDER_WIDTH, a_R, 
                     0, 255, 1, "%.0Lf", cvui.TRACKBAR_DISCRETE, 1) or
       cvui.trackbar(frame2, 7*PADX+SLIDER_WIDTH, 5*PADY+SLIDER_HEIGHT, 
                     SLIDER_WIDTH, a_G, 0, 255, 1, "%.0Lf", 
                     cvui.TRACKBAR_DISCRETE, 1) or
       cvui.trackbar(frame2, 10*PADX+2*SLIDER_WIDTH, 5*PADY+SLIDER_HEIGHT, 
                     SLIDER_WIDTH, a_B, 0, 255, 1, "%.0Lf", 
                     cvui.TRACKBAR_DISCRETE, 1)):
           tipo_str = "RGB sphere" 

    # RGB slicing
    cvui.text(frame2, PADX, 9*PADY+2*SLIDER_HEIGHT, f"R min: ")
    cvui.text(frame2, PADX*6+SLIDER_WIDTH, 9*PADY+2*SLIDER_HEIGHT, f"R max: ")
    cvui.text(frame2, PADX, 11*PADY+3*SLIDER_HEIGHT, f"G min: ")
    cvui.text(frame2, PADX*6+SLIDER_WIDTH, 11*PADY+3*SLIDER_HEIGHT, f"G max: ")
    cvui.text(frame2, PADX, 13*PADY+4*SLIDER_HEIGHT, f"B min: ")
    cvui.text(frame2, PADX*6+SLIDER_WIDTH, 13*PADY+4*SLIDER_HEIGHT, f"B max: ")
    if(cvui.trackbar(frame2, 5*PADX, 7*PADY+2*SLIDER_HEIGHT, 
                     SLIDER_WIDTH, R_min, 0, 255, 1, "%.0Lf", 
                     cvui.TRACKBAR_DISCRETE, 1) or
       cvui.trackbar(frame2, 10*PADX+SLIDER_WIDTH, 7*PADY+2*SLIDER_HEIGHT, 
                     SLIDER_WIDTH, R_max, 0, 255, 1, "%.0Lf", 
                     cvui.TRACKBAR_DISCRETE, 1) or
       cvui.trackbar(frame2, 5*PADX, 9*PADY+3*SLIDER_HEIGHT, 
                     SLIDER_WIDTH, G_min, 0, 255, 1, "%.0Lf", 
                     cvui.TRACKBAR_DISCRETE, 1) or
       cvui.trackbar(frame2, 10*PADX+SLIDER_WIDTH, 9*PADY+3*SLIDER_HEIGHT, 
                     SLIDER_WIDTH, G_max, 0, 255, 1, "%.0Lf", 
                     cvui.TRACKBAR_DISCRETE, 1) or
       cvui.trackbar(frame2, 5*PADX, 11*PADY+4*SLIDER_HEIGHT, 
                     SLIDER_WIDTH, B_min, 0, 255, 1, "%.0Lf", 
                     cvui.TRACKBAR_DISCRETE, 1) or
       cvui.trackbar(frame2, 10*PADX+SLIDER_WIDTH, 11*PADY+4*SLIDER_HEIGHT, 
                     SLIDER_WIDTH, B_max, 0, 255, 1, "%.0Lf", 
                     cvui.TRACKBAR_DISCRETE, 1)):
           tipo_str = "RGB cube" 

    # HSV slicing
    cvui.text(frame2, PADX, 15*PADY+5*SLIDER_HEIGHT, f"H min: ")
    cvui.text(frame2, PADX*6+SLIDER_WIDTH, 15*PADY+5*SLIDER_HEIGHT, f"H max: ")
    cvui.text(frame2, PADX, 17*PADY+6*SLIDER_HEIGHT, f"S min: ")
    cvui.text(frame2, PADX*6+SLIDER_WIDTH, 17*PADY+6*SLIDER_HEIGHT, f"S max: ")
    cvui.text(frame2, PADX, 19*PADY+7*SLIDER_HEIGHT, f"V min: ")
    cvui.text(frame2, PADX*6+SLIDER_WIDTH, 19*PADY+7*SLIDER_HEIGHT, f"V max: ")
    if(cvui.trackbar(frame2, 5*PADX, 13*PADY+5*SLIDER_HEIGHT, 
                     SLIDER_WIDTH, H_min, 0, 180, 1, "%.0Lf", 
                     cvui.TRACKBAR_DISCRETE, 1) or
       cvui.trackbar(frame2, 10*PADX+SLIDER_WIDTH, 13*PADY+5*SLIDER_HEIGHT, 
                     SLIDER_WIDTH, H_max, 0, 180, 1, "%.0Lf", 
                     cvui.TRACKBAR_DISCRETE, 1) or
       cvui.trackbar(frame2, 5*PADX, 15*PADY+6*SLIDER_HEIGHT, 
                     SLIDER_WIDTH, S_min, 0, 255, 1, "%.0Lf", 
                     cvui.TRACKBAR_DISCRETE, 1) or
       cvui.trackbar(frame2, 10*PADX+SLIDER_WIDTH, 15*PADY+6*SLIDER_HEIGHT, 
                     SLIDER_WIDTH, S_max, 0, 255, 1, "%.0Lf", 
                     cvui.TRACKBAR_DISCRETE, 1) or
       cvui.trackbar(frame2, 5*PADX, 17*PADY+7*SLIDER_HEIGHT, 
                     SLIDER_WIDTH, V_min, 0, 255, 1, "%.0Lf", 
                     cvui.TRACKBAR_DISCRETE, 1) or
       cvui.trackbar(frame2, 10*PADX+SLIDER_WIDTH, 17*PADY+7*SLIDER_HEIGHT, 
                     SLIDER_WIDTH, V_max, 0, 255, 1, "%.0Lf", 
                     cvui.TRACKBAR_DISCRETE, 1)):
           tipo_str = "HSV cube" 

    cvui.update(WINDOW_NAME_2)
    cvui.imshow(WINDOW_NAME_2, frame2)

    cvui.update(WINDOW_NAME_3)
    cvui.imshow(WINDOW_NAME_3, cv2.cvtColor(histograms, cv2.COLOR_RGBA2BGR))

    cvui.context(WINDOW_NAME_1)
    imagen = video.read()[1]
    img_bgr = np.copy(imagen)
    img_hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    hist_bgr = hi.get_histogram(img_bgr, mode="BGR")
    hist_hsv = hi.get_histogram(img_hsv, mode="HSV")
    if (tipo_str == "RGB sphere"):
       img_seg, mask = co.color_slicing_sphere(img_bgr, 
                                               [a_B[0], a_G[0], a_R[0]], 
                                               RGB_r[0])
       Rs, Gs, Bs = img_seg[mask,2], img_seg[mask,1], img_seg[mask,0]
       minRs, maxRs, minGs, maxGs, minBs, maxBs = 0, 0, 0, 0, 0, 0
       if len(Rs) > 0:
           minRs, maxRs = np.min(Rs), np.max(Rs)
       if len(Gs) > 0:
           minGs, maxGs = np.min(Gs), np.max(Gs)
       if len(Bs) > 0:
           minBs, maxBs = np.min(Bs), np.max(Bs)
       ax[0,0].clear()
       ax[0,0].stem(hist_bgr[2], markerfmt="", linefmt="r")
       ax[0,0].axvspan(minRs, maxRs, facecolor="r", alpha=0.3)
       ax[0,1].clear()
       ax[0,1].stem(hist_bgr[1], markerfmt="", linefmt="g")
       ax[0,1].axvspan(minGs, maxGs, facecolor="g", alpha=0.3)
       ax[0,2].clear()
       ax[0,2].stem(hist_bgr[0], markerfmt="", linefmt="b")
       ax[0,2].axvspan(minBs, maxBs, facecolor="b", alpha=0.3)
       fig.canvas.draw()
       histograms = np.array(fig.canvas.renderer._renderer)
    elif (tipo_str == "RGB cube"):
       img_seg, mask = co.color_slicing(img_bgr, 
                                        [B_min[0], G_min[0], R_min[0]], 
                                        [B_max[0], G_max[0], R_max[0]])
       ax[0,0].clear()
       ax[0,0].stem(hist_bgr[2], markerfmt="", linefmt="r")
       ax[0,0].axvspan(R_min[0], R_max[0], facecolor="r", alpha=0.3)
       ax[0,1].clear()
       ax[0,1].stem(hist_bgr[1], markerfmt="", linefmt="g")
       ax[0,1].axvspan(G_min[0], G_max[0], facecolor="g", alpha=0.3)
       ax[0,2].clear()
       ax[0,2].stem(hist_bgr[0], markerfmt="", linefmt="b")
       ax[0,2].axvspan(B_min[0], B_max[0], facecolor="b", alpha=0.3)
       fig.canvas.draw()
       histograms = np.array(fig.canvas.renderer._renderer)
    elif (tipo_str == "HSV cube"):
       img_seh, mask = co.color_slicing(img_hsv, 
                                        [H_min[0], S_min[0], V_min[0]], 
                                        [H_max[0], S_max[0], V_max[0]])
       img_seg = cv2.cvtColor(img_seh, cv2.COLOR_HSV2BGR)
       ax[1,0].clear()
       ax[1,0].stem(hist_hsv[0],markerfmt="",linefmt="darkorange")
       ax[1,0].axvspan(H_min[0],H_max[0],facecolor="darkorange",alpha=0.3)
       ax[1,1].clear()
       ax[1,1].stem(hist_hsv[1],markerfmt="",linefmt="limegreen")
       ax[1,1].axvspan(S_min[0],S_max[0],facecolor="limegreen",alpha=0.3)
       ax[1,2].clear()
       ax[1,2].stem(hist_hsv[2],markerfmt="",linefmt="royalblue")
       ax[1,2].axvspan(V_min[0],V_max[0],facecolor="royalblue",alpha=0.3)
       fig.canvas.draw()
       histograms = np.array(fig.canvas.renderer._renderer)
    frame1[:]=(49, 52, 49)
    cvui.text(frame1, PADX, PADY*2+ISHY, f"Imagen", 0.5)
    cvui.text(frame1, PADX*2+ISHX, PADY*2+ISHY, f"Mascara", 0.5)
    cvui.text(frame1, PADX*3+2*ISHX, PADY*2+ISHY, f"Segmentada", 0.5)
    cvui.image(frame1, PADX, PADY, imagen)
    cvui.image(frame1, PADX*2+ISHX, PADY, 
               cv2.cvtColor(255*mask.astype(np.uint8), cv2.COLOR_GRAY2BGR))
    cvui.image(frame1, PADX*3+2*ISHX, PADY, img_seg)

    cvui.update(WINDOW_NAME_1)
    cvui.imshow(WINDOW_NAME_1, frame1)

