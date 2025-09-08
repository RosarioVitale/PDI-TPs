"""
Ejercicio 3: Operaciones aritméticas
2. A partir de un video (pedestrians.mp4) de una cámara de seguridad, debe
obtener solamente el fondo de la imagen. Incorpore un elemento TrackBar
que le permita ir eligiendo el número de frames a promediar para observar
los resultados instantáneamente.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("../")
from utils import operations as ut
from cvui import cvui

IMAGE_DIR = "../imgs/"
IMAGE_FILE = "pedestrians.mp4"
cap = cv2.VideoCapture(f"{IMAGE_DIR}{IMAGE_FILE}")
imgs = []
print("Reading video...")
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        imgs.append(frame)
    else:
        break
cap.release()
print("Done. Read", len(imgs), "frames.")

# a) Promedio
avg_img = ut.suma_promedio(imgs)

def updateAvg(val):
    global avg_img, imgs, cant
    cant = val
    avg_img = ut.suma_promedio(imgs[:cant+1])

cant = len(imgs)
cv2.namedWindow("Pedestrians")
cv2.imshow("Pedestrians", avg_img)
cv2.createTrackbar("Frames", "Pedestrians", cant, cant, updateAvg)

while True:
    cv2.imshow("Pedestrians", avg_img)
    key = cv2.waitKey(20)
    if key == 27:
        break
