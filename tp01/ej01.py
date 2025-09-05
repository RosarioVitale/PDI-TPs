"""
Ejercicio 1: Lectura, visualización y escritura de imágenes.
1. Realice la carga y visualización de diferentes imágenes.
2. Muestre en pantalla información sobre las imágenes.
3. Investigue los formatos la imagen y como leer y como escribir un valor
puntual de la imagen.
4. Utilice el pasaje por parámetros para especificar la imagen a cargar.
5. Defina y recorte una subimagen de una imagen (vea ROI, Region Of Interest).
6. Investigue y realice una función que le permita mostrar varias imágenes en
una sóla ventana.
7. Dibuje sobre la imagen lı́neas, cı́rculos y rectángulos (opcional: defina la
posición en base al click del mouse).
"""

import cv2
import numpy as np

IMG_PATH = "../imgs/"
img1 = cv2.imread(f"{IMG_PATH}cameraman.tif", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(f"{IMG_PATH}clown.jpg", cv2.IMREAD_GRAYSCALE)
img3 = cv2.imread(f"{IMG_PATH}coins.tif", cv2.IMREAD_GRAYSCALE)
img4 = cv2.imread(f"{IMG_PATH}parrot.tif", cv2.IMREAD_GRAYSCALE)

cv2.namedWindow("imagen")
inix, iniy, imgsize = 0, 0, 256
img4_roi = img4[inix:inix+imgsize, iniy:iniy+imgsize].copy()
full_img = np.zeros((imgsize*2, imgsize*2), np.uint8)

full_img[0:imgsize, 0:imgsize] = img1
full_img[0:imgsize, imgsize:imgsize*2] = img2
full_img[imgsize:imgsize*2, 0:imgsize] = img3
full_img[imgsize:imgsize*2, imgsize:imgsize*2] = img4_roi[0:imgsize, 0:imgsize]

def on_trackbar(val):
    global img4, img4_roi, full_img, inix, iniy
    inix = cv2.getTrackbarPos("X", "imagen")
    iniy = cv2.getTrackbarPos("Y", "imagen")
    img4_roi = img4[inix:inix+imgsize, iniy:iniy+imgsize].copy()
    full_img[imgsize:imgsize*2, imgsize:imgsize*2] = img4_roi

cv2.createTrackbar("X", "imagen", 0, img4.shape[0]-imgsize, on_trackbar)
cv2.createTrackbar("Y", "imagen", 0, img4.shape[1]-imgsize, on_trackbar)

while True:
    cv2.imshow("imagen", full_img)
    if cv2.waitKey(1) == ord("q"):
        break
