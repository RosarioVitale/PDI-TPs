import numpy as np

#########
#  LUT  #
#########
def computeLUT(R, a, c, smin=0, smax=255, dtype="uint8"):
    """Aplicamos LUT dada la tabla de intensidades R y los coeficientes a y c.

    Parameters:
        R: Tabla de valores a mapear.
        a: Factor de ganancia.
        c: Offset.
        smin: Minimo a partir del cual se aceptan los valores resultantes. Por
        defecto es 0.
        smax: Maximo hasta el cual se aceptan los valores resultantes. Por
        defecto es 255.
        dtype: Tipo de dato de la imagen resultante ("uint8" por defecto)
    Returns:
        S: Tabla de intensidades resultante.

    """
    S = a*R.astype("float")+c
    S = np.clip(S, smin, smax)
    return S.astype(dtype)


def applyLUT(imagen, A, C, Rlims=[(0, 255)], rstep=1, smin=0, smax=255,
             dtype="uint8"):
    """Aplicamos LUT a la imagen dada. Se reciben los valores de los
    coeficientes en listas, para pasarse por tramos.

    Parameters:
        imagen: Imagen a transformar.
        A: Lista de factores de ganancia.
        C: Lista de offsets.
        Rlims: Lista de tuplas de la forma [(rmin1, rmax1), (rmin2, rmax2),
        ...] para determinar los limites de los tramos. Por defecto cubre todo
        el rango asi se hace un solo tramo.
        rstep: Paso para generar la tabla de intensidades. Por defecto es 1.
        smin: Minimo a partir del cual se aceptan los valores resultantes. Por
        defecto es 0.
        smax: Maximo hasta el cual se aceptan los valores resultantes. Por
        defecto es 255.
        dtype: Tipo de dato de la imagen resultante ("uint8" por defecto)
    Returns:
        new_image: Imagen resultante.

    """
    new_image = np.zeros_like(imagen, dtype=dtype)
    for i in range(len(Rlims)):
        if Rlims[i][0] != Rlims[i][1]:
            R = np.arange(Rlims[i][0], Rlims[i][1]+rstep, rstep, dtype=dtype)
            S = computeLUT(R, A[i], C[i], smin, smax, dtype)
            for j in range(len(R)):
                new_image[imagen==R[j]] = S[j]
    return new_image


def applyNegative(imagen):
    """Aplicamos el negativo (simplemente llamamos a applyLUT pero con los
    parametros adecuados)

    Parameters:
        imagen: Imagen a transformar.
    Returns:
        new_image: Imagen resultante.

    """
    return applyLUT(imagen, [-1], [np.max(imagen)])


def plotLUT(ax, A, C, Rlims=[(0, 255)], rstep=1, rmin=0, rmax=255, smin=0,
            smax=255):
    """Graficamos el mapeo (es decir la recta de la funcion aplicada)

    Parameters:
        ax: Axis para graficar.
        A: Lista de factores de ganancia.
        C: Lista de offsets.
        Rlims: Lista de tuplas de la forma [(rmin1, rmax1), (rmin2, rmax2),
        ...] para determinar los limites de los tramos. Por defecto cubre todo
        el rango asi se hace un solo tramo.
        rstep: Paso para generar la tabla de intensidades. Por defecto es 1.
        rmin: Limite inferior en x. Por defecto es 0.
        rmax: Limite superior en x. Por defecto es 255.
        smin: Limite inferior en y. Por defecto es 0.
        smax: Limite superior en y. Por defecto es 255.
    Returns:
        ax: Axis modificados.

    """
    for i in range(len(Rlims)):
        if Rlims[i][0] != Rlims[i][1]:
            R = np.arange(Rlims[i][0], Rlims[i][1]+rstep, rstep)
            ax.plot(R, computeLUT(R, A[i], C[i], smin, smax), 'b-',
                    linewidth=2)
        else:
            # Esto dibuja linea vertical. Como A y C deben tener el mismo
            # tamano que Rlims, aprovecho esos lugares para saber desde donde
            # y hasta donde hacer la linea
            ax.plot([Rlims[i][0], Rlims[i][1]], [a[i], c[i]], 'b-',
                    linewidth=2)
    ax.set(xlim=(rmin, rmax+rstep), ylim=(smin, smax+rstep))
    return ax


def hacerTramos(points):
    """Recibe los puntos necesarios para generar los tramos de la LUT. Fija
    los puntos (0, 0) y (255, 255)

    Parameters:
        points: Lista de tuplas de la forma [(x1, y1), (x2, y2), ...] para
        determinar por donde pasa la funcion por tramos.
    Returns:
        A: Lista de factores de ganancia.
        C: Lista de offsets.
        Rlims: Lista de tuplas de la forma [(rmin1, rmax1), (rmin2, rmax2),
        ...] para determinar los limites de los tramos.

    """
    p0 = (0,0)
    A, C, Rlims = [], [], []
    for p in points:
        # TODO: Agregar linea vertical o warning si el denominador es 0 #
        if abs(p[0]-p0[0])>1e-3:
            a = (p[1]-p0[1])/(p[0]-p0[0])
            c = p[1]-a*p[0]
        else:
            # no deberian accederse nunca a estos valores, salvo para plot
            a, c = p0[1], p[1]
        A.append(a)
        C.append(c)
        Rlims.append((p0[0], p[0]))
        p0=p
    p=(255, 255)
    a = (p[1]-p0[1])/(p[0]-p0[0])
    c = p[1]-a*p[0]
    A.append(a)
    C.append(c)
    Rlims.append((p0[0], p[0]))
    return A, C, Rlims


#################
#  No lineales  #
#################
def computeLog(R, c=1, smin=0, smax=1, dtype="uint8"):
    """Aplicamos transformacion logaritmica a la entrada.

    Parameters:
        R: Tabla de valores (o imagen)
        c: Constante de escalado???
        dtype: Tipo de dato de la imagen resultante ("uint8" por defecto)
    Returns:
        S: Tabla de intensidades resultante.

    """
    if dtype=="uint8":
        R= R.astype("float")/255.0
    S = c*np.log(1.+R, dtype="float")
    S = np.clip(S, smin, smax)
    if dtype=="uint8":
        S*=255
    return S.astype(dtype)


def computePow(R, gamma, c=1, smin=0, smax=1., dtype="uint8"):
    """Aplicamos transformacion de potencia a la entrada.

    Parameters:
        R: Tabla de valores (o imagen)
        gamma: Exponente gamma.
        c: Constante de escalado???
        dtype: Tipo de dato de la imagen resultante ("uint8" por defecto)
    Returns:
        S: Tabla de intensidades resultante.

    """
    if dtype=="uint8":
        R= R.astype("float")/255.0
    S = c*np.power(R, gamma, dtype="float")
    S = np.clip(S, smin, smax)
    if dtype=="uint8":
        S*=255
    return S.astype(dtype)


#################
#  Aritmeticas  #
#################
def blending(img1, img2, a, imax=255, imin=0):
    """Aplicamos alpha blending dadas dos imagenes y el alpha.

    Parameters:
        img1: Primera imagen a sumar.
        img2: Segunda imagen a sumar.
        a: Valor de alpha.
        imin: Minimo a partir del cual se aceptan los valores resultantes. Por
        defecto es 0.
        imax: Maximo hasta el cual se aceptan los valores resultantes. Por
        defecto es 255.
    Returns:
        res: Imagen resultante.

    """
    img = (1-a)*img1+a*img2
    img = np.clip(img, imin, imax)
    return img


def suma(img1, img2, imax=255, imin=0):
    """Sumamos dos imagenes dadas.

    Parameters:
        img1: Primera imagen a sumar.
        img2: Segunda imagen a sumar.
        imin: Minimo a partir del cual se aceptan los valores resultantes. Por
        defecto es 0.
        imax: Maximo hasta el cual se aceptan los valores resultantes. Por
        defecto es 255.
    Returns:
        res: Imagen resultante.

    """
    img = img1+img2
    img = np.clip(img, imin, imax)
    return img


def suma_promedio(imgs, imin=0, imax=255, dtype="uint8"):
    """Calculamos el promedio de varias imagenes.

    Parameters:
        imgs: Lista de imagenes a promediar.
        imin: Minimo a partir del cual se aceptan los valores resultantes. Por
        defecto es 0.
        imax: Maximo hasta el cual se aceptan los valores resultantes. Por
        defecto es 255.
        dtype: Tipo de dato de la imagen resultante ("uint8" por defecto)
    Returns:
        img: Imagen resultante.

    """
    img = imgs[0].astype("float")
    for i in range(1, len(imgs)):
        img += imgs[i].astype("float")

    img /= len(imgs)
    img = np.clip(img, imin, imax)
    return img.astype(dtype)


def diferencia(img1, img2, reesc="sum", imin=0, imax=255, dtype="uint8"):
    """Calculamos la resta de dos imagenes y reescalamos aplicando el metodo
    seleccionado.

    Parameters:
        img1: Primera imagen a restar.
        img2: Segunda imagen a restar.
        reesc: Metodo de reescalado. Si es "sum", suma 255 y divide por 2. Si
        es "res", resta el minimo y escala a 255 ("sum" por defecto)
        dtype: Tipo de dato de la imagen resultante ("uint8" por defecto)
    Returns:
        img: Imagen resultante.

    """
    img1=img1.astype("float")
    img2=img2.astype("float")
    img = img1-img2
    if reesc == "sum":
        img = (img+255)/2
    elif reesc == "res":
        img = (img-img.min())*(255/(img.max()-img.min()))
    elif reesc == "no":
        img = np.clip(img, imin, imax)
    else:
        print("Error: elegir uno de los valores para reescalar: sum o res")
        return

    return img.astype(dtype)


def multiplicacion(img, mask, dtype="uint8"):
    """Multiplicamos una imagen por una mascara binaria.

    Parameters:
        img: Imagen a multiplicar.
        mask: Mascara binaria a aplicar.
        dtype: Tipo de dato de la imagen resultante ("uint8" por defecto)
    Returns:
        res: Imagen resultante.

    """
    img=img.astype("float")
    if np.max(mask)>1:
        mask = mask/np.max(mask)
    res = img*mask
    return res.astype(dtype)


