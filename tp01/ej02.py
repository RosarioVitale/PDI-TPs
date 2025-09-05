"""
Ejercicio 2: Información de intensidad.
1. Informe los valores de intensidad de puntos particulares de la imagen
(opcional: determine la posición en base al click del mouse).
2. Obtenga y grafique los valores de intensidad (perfil de intensidad ) sobre
una determinada fila o columna.
3. Grafique el perfil de intensidad para un segmento de interés cualquiera.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_column(img, col):
    return img[:, col]

def get_row(img, row):
    return img[row, :]

def get_segment(img, x1, y1, x2, y2):
    if x1==x2 and y1==y2:
        return img[x1, y1]
    if x1==x2:
        return img[y1:y2, x1]
    if y1==y2:
        return img[y1, x1:x2]
    Y = [int(x*(y2-y1)//(x2-x1)+y1) for x in range(x2-x1)]
    X = [i for i in range(x1,x2)] 
    return img[Y,X]

def mark_profiles(img, x, y, seg):
    #print(f"Marking {x}, {y}, {seg}")
    dst = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # draw selected row in red
    dst = cv2.line(dst, (0, y), (img.shape[1], y), (0, 0, 255), 2)
    # draw selected column in green
    dst = cv2.line(dst, (x, 0), (x, img.shape[0]), (0, 150, 0), 2)
    # draw selected segment in blue
    dst = cv2.line(dst, (seg[0], seg[1]), (seg[2], seg[3]), (255, 0, 0), 2)
    return dst

def update(sel_pnt, sel_seg):
    global window, fig, ax
    x, y = sel_pnt
    col = get_column(img, x)
    row = get_row(img, y)
    seg = get_segment(img, sel_seg[0], sel_seg[1], sel_seg[2], sel_seg[3])
    ax[0].clear()
    ax[0].plot(row, color="red")
    ax[0].set_title("row")
    ax[1].clear()
    ax[1].plot(col, color="green")
    ax[1].set_title("column")
    ax[2].clear()
    ax[2].plot(seg, color="blue")
    ax[2].set_title("segment")
    fig.canvas.draw()
    fig_img = np.array(fig.canvas.renderer._renderer)
    window = np.zeros((fig_img.shape[0]+img.shape[0], fig_img.shape[1], 3), 
                      dtype=np.uint8)
    window[0:fig_img.shape[0],0:fig_img.shape[1],:] = fig_img[:,:,[2,1,0]]
    window[fig_img.shape[0]:,(fig_img.shape[1]-img.shape[1])//2:
           (fig_img.shape[1]+img.shape[1])//2,:] = mark_profiles(img, 
                                                                 sel_pnt[0], 
                                                                 sel_pnt[1], 
                                                                 sel_seg)
    cv2.imshow("Hello World", window)

def on_trackbar(val):
    x1 = cv2.getTrackbarPos("X1", "Hello World")
    y1 = cv2.getTrackbarPos("Y1", "Hello World")
    x2 = cv2.getTrackbarPos("X2", "Hello World")
    y2 = cv2.getTrackbarPos("Y2", "Hello World")
    if x1<x2:
        seg = (x1, y1, x2, y2)
    else:
        seg = (x2, y2, x1, y1)
    global sel_seg, sel_pnt
    sel_seg = seg
    update(sel_pnt, seg)

def on_click(event, x, y, flags, param):
    global fig_img
    if event == cv2.EVENT_LBUTTONDOWN:
        if (y>fig_img.shape[0] and x>(fig_img.shape[1]-img.shape[1])//2 and 
            x<(fig_img.shape[1]+img.shape[1])//2): 
            global sel_pnt, sel_seg
            x = x-(fig_img.shape[1]-img.shape[1])//2
            y = y-fig_img.shape[0]
            sel_pnt = (x, y)
            update(sel_pnt, sel_seg)

img = cv2.imread("../imgs/snowman.png", cv2.IMREAD_GRAYSCALE)
sel_pnt = (img.shape[1]//2, img.shape[0]//2)
sel_seg = (img.shape[1]//4, img.shape[0]//4, img.shape[1]*3//4, 
           img.shape[0]*3//4)
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
col = get_column(img, sel_pnt[0])
row = get_row(img, sel_pnt[1])
seg = get_segment(img, sel_seg[0], sel_seg[1], sel_seg[2], sel_seg[3])
ax[0].plot(row, color="red")
ax[0].set_title("row")
ax[1].plot(col, color="green")
ax[1].set_title("column")
ax[2].plot(seg, color="blue")
ax[2].set_title("segment")

fig.canvas.draw()
fig_img = np.array(fig.canvas.renderer._renderer)
window = np.zeros((fig_img.shape[0]+img.shape[0], fig_img.shape[1], 3), 
                  dtype=np.uint8)
window[0:fig_img.shape[0], 0:fig_img.shape[1], :] = fig_img[:, :, [2, 1, 0]]
window[fig_img.shape[0]:, (fig_img.shape[1]-img.shape[1])//2:
       (fig_img.shape[1]+img.shape[1])//2, :] = mark_profiles(img, sel_pnt[0], 
                                                              sel_pnt[1], 
                                                              sel_seg)

cv2.namedWindow("Hello World")
cv2.setMouseCallback("Hello World", on_click)
cv2.createTrackbar("X1", "Hello World", img.shape[1]//4, img.shape[1], 
                   on_trackbar)
cv2.createTrackbar("Y1", "Hello World", img.shape[0]//4, img.shape[0], 
                   on_trackbar)
cv2.createTrackbar("X2", "Hello World", img.shape[1]*3//4, img.shape[1], 
                   on_trackbar)
cv2.createTrackbar("Y2", "Hello World", img.shape[0]*3//4, img.shape[0], 
                   on_trackbar)
while True:
    cv2.imshow("Hello World", window)
    if cv2.waitKey(1) == ord("q"):
        break
