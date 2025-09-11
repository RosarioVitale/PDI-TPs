"""
Ejercicio 2: Trabajos de aplicación
2. Realice un algoritmo de búsqueda por correlación de histogramas de 
intensidad. Se debe informar el contenido de la imagen: Bandera, Caricatura, 
Personaje o Paisaje. Utilice las imágenes disponibles en 
Busqueda_histograma.zip.
"""
import cv2
import numpy as np
import os
import sys
sys.path.append("../")
from cvui import cvui
from utils import utils, operations


def load_image(NIMG, IMAGE_DIR):
    """Cargamos imagen dado el numero de la misma.

    Parameters:
        NIMG: ID de la imagen.
        IMAGE_DIR: Directorio con imagenes guardadas.
    Returns:
        imagen: Imagen cargada y procesada.

    """
    imgname = f"{IMAGE_DIR}imagen{NIMG:02d}"
    filename = imgname+(".jpg" if os.path.isfile(imgname+".jpg") else ".JPG")
    imagen = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    imagen = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
    return imagen, filename


def get_sl_data(imagen):
    """Obtenemos histograma.

    Parameters:
        imagen: TODO
    Returns:
        TODO

    """
    return cv2.calcHist([imagen], [0], None, [256], [0, 256])


def get_prediction(hist, refs):
    """TODO: Docstring for get_prediction.

    Parameters:
        hist: TODO
        refs: TODO
    Returns:
        TODO

    """
    pred = "?????"
    max_corr=-1
    for label, rhist in refs.items():
        corr = cv2.compareHist(hist, rhist, cv2.HISTCMP_CORREL)
        if corr>max_corr:
            pred = label
            max_corr = corr

    return pred


labels = ["Bandera", "Caricatura", "Paisaje", "Personaje"]
refs = {}
for i in range(len(labels)):
    rimg, _ = load_image(i+1, "busq_hist/")
    refs[labels[i]] = get_sl_data(rimg)


WINDOW_NAME = "Busqueda por correlacion"
cvui.init(WINDOW_NAME)

NIMG = 1
IMAGE_DIR = "busq_hist/"
imagen, filename = load_image(NIMG, IMAGE_DIR)
width = 600
height = 320
max_nimg = len([f for f in os.listdir(IMAGE_DIR) if "imagen" in f])
frame = np.zeros((height, width, 3), np.uint8)

spvals=get_sl_data(imagen)
pred_str = get_prediction(spvals, refs)

while True:
    frame[:] = (49, 52, 49)
    cvui.beginRow(frame, 10, 10, -1, -1, 20)
    if (cvui.button("Prev")):
        # The button was clicked, so let's decrement our counter.
        NIMG -= 1
        if NIMG < 1:
            NIMG = 1
        else:
            imagen, filename = load_image(NIMG, IMAGE_DIR)
    cvui.text(filename)
    if (cvui.button("Next")):
        # The button was clicked, so let's increment our counter.
        NIMG += 1
        if NIMG > max_nimg:
            NIMG = max_nimg
        else:
            imagen, filename = load_image(NIMG, IMAGE_DIR)
    cvui.text(f"    Class: {pred_str}")
    cvui.endRow()

    cvui.beginRow(frame, 20, 50, -1, -1)
    cvui.image(imagen)
    spvals=get_sl_data(imagen)
    cvui.sparkline(frame, spvals, 300, 55, 255, 250)
    pred_str = get_prediction(spvals, refs)
    cvui.endRow()

    # Update cvui internal stuff
    cvui.update()

    # Show window content
    cvui.imshow(WINDOW_NAME, frame)

    # Press ESC to exit
    if cv2.waitKey(20) == 27:
        break
