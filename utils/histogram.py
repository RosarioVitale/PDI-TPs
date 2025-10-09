import cv2


def get_histogram(img, mode="GRAY", mask=None, bins=None, rang=None):
    """
    Funcion para obtener el histograma de una imagen.

    Parameters:
    img: Imagen
    mode: Modo de la imagen (GRAY, BGR, HSV)
    mask: Mascara de la imagen para seleccionar regiones (por defecto toma los
        pixeles de toda la imagen)
    bins: Numero de bins del histograma (por defecto 256). Si la imagen es en
    color, debe ser una lista de 3 enteros con el numero de bins de cada canal.
    rang: Rango del histograma (por defecto [0, 256]). Si la imagen es en
    color, debe ser una lista de 3 listas con el rango de cada canal.

    Returns:
    hist: Histograma

    """
    if mode=="GRAY":
        if bins is None:
            bins = 256
        if rang is None:
            rang=[0, 256]
        hist = cv2.calcHist([img], [0], mask, [bins], rang)
    elif mode=="BGR":
        bgr_planes = cv2.split(img)
        if bins is None:
            bins = [256, 256, 256]
        if rang is None:
            rang=[[0, 256], [0, 256], [0, 256]]
        hist = [cv2.calcHist(bgr_planes, [0], mask, [bins[0]], rang[0]),
                cv2.calcHist(bgr_planes, [1], mask, [bins[1]], rang[1]),
                cv2.calcHist(bgr_planes, [2], mask, [bins[2]], rang[2])]
    elif mode=="HSV":
        hsv_planes = cv2.split(img)
        if bins is None:
            bins = [181, 256, 256]
        if rang is None:
            rang=[[0, 181], [0, 256], [0, 256]]
        hist = [cv2.calcHist(hsv_planes, [0], mask, [bins[0]], rang[0]),
                cv2.calcHist(hsv_planes, [1], mask, [bins[1]], rang[1]),
                cv2.calcHist(hsv_planes, [2], mask, [bins[2]], rang[2])]
    else:
        print("ERROR! Seleccione un modo valido")
    return hist


def equalize_hist(img, mode="GRAY", channels=None):
    """
    Funcion para equalizar el histograma de una imagen

    Parameters:
    img: Imagen
    mode: Modo de la imagen (GRAY, BGR, HSV)
    channels: Listas de indices de los canales a equalizar (por defecto toma 
    todos los canales, es decir, [0, 1, 2]).
    Returns:
    img_eq: Imagen equalizada

    """
    if mode=="GRAY":
        img_eq = cv2.equalizeHist(img)
    elif mode=="BGR" or mode=="HSV":
        planes = cv2.split(img)
        if channels is None:
            channels = [0, 1, 2]
        pln_eq = []
        for c in range(len(planes)):
            if c in channels:
                pln_eq.append(cv2.equalizeHist(planes[c]))
            else:
                pln_eq.append(planes[c])
        img_eq=cv2.merge(pln_eq)
    else:
        print("ERROR! Seleccione un modo valido")
    return img_eq


def mean(hist):
    Prob = hist/sum(hist)
    mean = 0
    for g in range(hist.shape[0]):
        mean += g * Prob[g]
    return mean


def variance(hist):
    Prob = hist/sum(hist)
    mean = media(hist)
    variance = 0
    for g in range(hist.shape[0]):
        variance += (g - mean)**2 * Prob[g]
    return variance


def skewness(hist):
    Prob = hist/sum(hist)
    mean = media(hist)
    skewness = 0
    for g in range(hist.shape[0]):
        skewness += (g - mean)**3 * Prob[g]
    return skewness


def energy(hist):
    Prob = hist/sum(hist)
    energy = 0
    for g in range(hist.shape[0]):
        energy += Prob[g]**2
    return energy


def entropy(hist):
    Prob = hist/sum(hist)
    entropy = 0
    for g in range(hist.shape[0]):
        if Prob[g] != 0:
            entropy += -Prob[g] * np.log2(Prob[g])
    return entropy



