"""
Ejercicio 3: Aplicación
Utilice las herramientas aprendidas en esta unidad para implementar un sistema
que permita identificar una botella que no está correctamente llena. Las
imágenes que se proporcionarán son capturadas con una cámara fija, en escala de 
grises y directamente de la lı́nea de envasado. Para implementar el sistema 
deberá bastarle una imagen de ejemplo “botella.tif” (que encontrará en el 
repositorio). Adicionalmente, se espera que el sistema pueda:
- identificar una botella no-llena en cualquier posición de la imagen.
- indicar la posición de la botella en la imagen (podrı́a ser con un recuadro,
- informando la posición relativa entre botellas, la posición absoluta en
    pixels, etc).
- informar el porcentaje de llenado de la botella no-llena.
"""

import cv2
import numpy as np
import os

def get_column(img, col):
    return img[:, col]


def get_row(img, row):
    return img[row, :]


def detectar_botellas(img, th=10):
    """
    Detectar límites horizontales de botellas en la imagen
    """
    xlims = []
    row = get_row(img, img.shape[0]//2)
    background = np.where(row < th)[0]
    if (background[0] > 0):
        # Si el primer píxel no es de fondo, la primera botella aparece en el 
        # borde de la imagen. El primer elemento de background es el primer
        # píxel de fondo y con él se limita la primera botella
        xlims.append((0, background[0]))

    # Con lims se puede saber dónde hay saltos de fondo a botella y viceversa
    # (se ignoran saltos de menos de 5 píxeles ya que las botellas no van a 
    # tan pequenas)
    lims = np.where(background[1:] - background[:-1] > 5)[0]
    for i in range(len(lims)):
        xlims.append((background[lims[i]], background[lims[i]+1]))

    if (background[-1] < img.shape[1] and img.shape[1] - background[-1] > 5):
        # Si el último píxel no es de fondo, la última botella aparece en el 
        # borde de la imagen. El último elemento de background es el último
        # píxel de fondo y con ello se limita la última botella
        xlims.append((background[-1], img.shape[1]))
    return xlims


def detectar_alturas(img, xlims, th_liq=200):
    """
    Detectar límites verticales de llenado de botellas en la imagen. Para esto
    se puede aprovechar que ya se sabe la ubicación de las botellas y medir 
    las columnas centrales de las mismas
    """
    yliqs = []  # Alturas de llenado de botellas
    ymin = img.shape[0] # Altura de la botella completa (se asume que todas 
                        # tienen la misma altura)
    if len(xlims) == 0:
        return yliqs, ymin, ymin
    for lim in xlims:
        x = (lim[0] + lim[1]) // 2  # Centro de la botella detectada antes
        column = img[:-2, x]  # Ignoramos los dos ultimos pixeles porque
                              # corresponden a la cinta transportadora
        # Esta bandera me permite saber si el pixel actual corresponde a una
        # botella propiamente o no (con "no" se refiere a fondo y liquido)
        is_bottle = False
        ylev = len(column)  # Y donde se encuentra el liquido de la botella
                            # en la columna actual (inicialmente es el último
                            # píxel de la columna para indicar que la botella
                            # no tiene liquido)
        for y in range(len(column)):
            if is_bottle and column[y] < th_liq:
                # Si el píxel anterior era botella y el actual tiene menos 
                # intensidad, entonces se ha llegado al liquido
                is_bottle = False
                ylev = y
            if not is_bottle and column[y] > th_liq:
                # Si el píxel anterior no era botella y el actual tiene mas
                # intensidad, entonces se ha llegado a la botella
                is_bottle = True
                if ymin > y:
                    ymin = y
        yliqs.append(ylev)

    ymax = len(column) # Último píxel de la botella completa (para envolver 
                       # las botellas con recuadros usando ymin, ymax y los 
                       # limites de xlims)

    return yliqs, ymin, ymax


def calcular_porcentaje_llenado(yliqs, ymin):
    """
    Calculamos los porcentajes de llenado a partir de comparar el punto mas
    alto. Asumimos que al menos una botella esta llena, para no tener que 
    fijarlo nosotros
    """
    if len(yliqs) == 0:
        return []
    ymax = min(yliqs)
    if np.abs(ymax - ymin) < 5:
        return [0 for y in yliqs]
    percs = [100*(ymin-y)/(ymin-ymax) for y in yliqs]
    return percs


IMG_PATH = "botellas/"
RES_PATH = "botellas_marcadas/"
for IMG_FILE in os.listdir(IMG_PATH):
    print(f"Procesando {IMG_FILE}")
    img = cv2.imread(f"{IMG_PATH}{IMG_FILE}", cv2.IMREAD_GRAYSCALE)

    xlims = detectar_botellas(img)
    ylims, min_y, max_y = detectar_alturas(img, xlims)
    percs = calcular_porcentaje_llenado(ylims, max_y)

    dst = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i in range(len(xlims)):
        x0, x1 = xlims[i]
        int_c = int(200*(percs[i]/100))
        color = (0, int_c, 255) if percs[i] < 100 else (0, 150, 0)
        dst = cv2.rectangle(dst, (x0, min_y), (x1, max_y), color, 2)
        dst = cv2.putText(dst, f"B{i+1}", (x0+5, min_y+15), 2, 0.5, color, 
                          1)
        dst = cv2.line(dst, (x0, ylims[i]), (x1, ylims[i]), color, 2)
        dst = cv2.putText(dst, f"{percs[i]:.2f}%", (x0+5, ylims[i]-5), 0, 0.35, 
                          color, 1)

    cv2.imwrite(f"{RES_PATH}{IMG_FILE}", dst)

