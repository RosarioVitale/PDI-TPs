import numpy as np
import cv2

######################
# Profiles functions #
######################
def get_column(img, col):
    """
    Devuelve la columna col de la imagen img

    Parameters:
        img: Imagen
        col: Columna a obtener

    Returns:
        column: Perfil de intensidad de la columna
    """
    return img[:, col]


def get_row(img, row):
    """
    Devuelve la fila row de la imagen img

    Parameters:
        img: Imagen
        row: Fila a obtener

    Returns:
        row: Perfil de intensidad de la fila
    """
    return img[row, :]


def get_segment(img, x1, y1, x2, y2):
    """
    Devuelve un segmento de la imagen img

    Parameters:
        img: Imagen
        x1: x inicial
        y1: y inicial
        x2: x final
        y2: y final

    Returns:
        segment: Segmento de la imagen
    """
    if x1==x2 and y1==y2:
        return img[x1, y1]
    if x1==x2:
        return img[y1:y2, x1]
    if y1==y2:
        return img[y1, x1:x2]
    Y = [int(x*(y2-y1)//(x2-x1)+y1) for x in range(x2-x1)]
    X = [i for i in range(x1,x2)] 
    return img[Y,X]


#################
# Gif functions #
#################
def load_gif(filename):
    """
    Devuelve las imagenes que componen un gif.

    Parameters:
        filename: Nombre del gif

    Returns:
        frames: Lista de imagenes. Si el gif solo tiene un frame, lo devuelve
    """
    aux = cv2.VideoCapture(filename)
    frames = []
    while aux.isOpened():
        ret, frame = aux.read()
        if ret:
            frames.append(frame)
        else:
            break
    aux.release()
    if len(frames) == 0:
        return None
    if len(frames) == 1:
        return frames[0]
    return frames

###################
# Error functions #
###################
def error_mse(img1, img2):
    """
    Calcula el error cuadratico medio entre dos imagenes.

    Parameters:
        img1: Imagen 1
        img2: Imagen 2

    Returns:
        error: Error cuadratico medio
    """
    return np.mean((img1.astype("float")-img2.astype("float"))**2)
