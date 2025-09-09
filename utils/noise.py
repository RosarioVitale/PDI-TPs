import numpy as np
import cv2

def add_impulsive_noise(image, Pa=0.1, Pb=0.1, A=0, B=255):
    """Agregamos ruido impulsivo a una imagen dada la proporcion de sal y
    pimienta deseada.

    Parameters:
        image: Imagen a la que se le va a agregar el ruido
        Pa: Proporcion de pimienta (0.1 por defecto)
        Pb: Proporcion de sal (0.1 por defecto)
        A: Valor de pimienta (0 por defecto)
        B: Valor de sal (255 por defecto)
    Returns:
        img_n: Imagen con ruido impulsivo

    """
    row, col = image.shape
    img_n = image.copy()
    n_rand = np.random.rand(row,col)
    for x in range(row):
        for y in range(col):
            if n_rand[x,y] <= Pa:
                img_n[x,y] = A #--> Pepper
            elif n_rand[x,y] <= Pa+Pb:
                img_n[x,y] = B #--> Salt
    return img_n
