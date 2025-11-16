import numpy as np
import cv2
from skimage.util.shape import view_as_windows
from scipy.stats.mstats import gmean, hmean
from scipy.stats import trim_mean

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


def add_gaussian_noise(image, mean=0, stddev=1):
    """Agregamos ruido gaussiano a una imagen.

    Parameters:
        image: Imagen a la que se le va a agregar el ruido
        mean: Media del ruido (0 por defecto)
        stddev: Desviacion estandar del ruido (1 por defecto)
    Returns:
        img_n: Imagen con ruido gaussiano

    """
    row, col = image.shape
    noise = np.zeros((row,col))
    noise = cv2.randn(noise, mean, stddev) #--> Gaussian distribution
    img_n = image.copy().astype("float") + noise
    #img_n = image.copy() + noise
    return img_n.astype("uint8")


def add_uniform_noise(image, A=108, B=148):
    """Agregamos ruido uniforme a una imagen.

    Parameters:
        image: Imagen a la que se le va a agregar el ruido
        A: Limite inferior (108 por defecto)
        B: Limite superior (148 por defecto)
    Returns:
        img_n: Imagen con ruido uniforme

    """
    row, col = image.shape
    noise = np.zeros((row,col))
    noise = cv2.randu(noise, A, B) #--> Uniform distribution
    noise = noise - ((A+B)/2)      #--> Convert to: Mean = 0
    img_n = image.copy().astype("float") + noise
    img_n = cv2.normalize(img_n,None,0,255,cv2.NORM_MINMAX)
    return img_n.astype("uint8")


def add_exponential_noise(image, a=0.05):
    """Agregamos ruido exponencial a una imagen.

    Parameters:
        image: Imagen a la que se le va a agregar el ruido
        a: Parametro de la distribucion exponencial (0.05 por defecto)
    Returns:
        img_n: Imagen con ruido exponencial

    """
    row, col = image.shape
    noise = np.zeros((row,col))
    noise = np.random.exponential(1/a, size=(row,col)) 
    img_n = image.copy().astype("float") + noise
    return img_n.astype("uint8")


#############################################################################
# Mean filters
#############################################################################
def windowize(image, m):
    """Separamos la imagen en ventanas de tamaño m x m (funcion auxiliar para 
    aplicar los filtros).

    Parameters:
        image: Imagen a ventanear
        m: Tamaño de la ventana
    Returns:
        img_windows: Ventanas de la imagen.

    """
    row, col = image.shape
    pad = m//2
    img_padded = cv2.copyMakeBorder(image, *[pad]*4, cv2.BORDER_REFLECT_101)
    img_windows = view_as_windows(img_padded, (m,m)) #--> split in windows
    return img_windows


def geometric_mean_filter(image, m=3):
    """Filtro de media geometrica

    Parameters:
        image: Imagen a la que se le va a aplicar el filtro
        m: Tamaño de las ventanas en las que aplicar el filtro
    Returns:
        img_f: Imagen filtrada

    """
    img_windows = windowize(image,m) #--> split in windows
    img_f = gmean(img_windows+0.0000001, axis=(2,3))
    if m%2==0: 
        img_f=img_f[0:-1,0:-1]
    return img_f.astype("uint8")


def contraharmonic_mean_filter(image, m=3, Q=2):
    """Filtro de media contraharmonica

    Parameters:
        image: Imagen a la que se le va a aplicar el filtro
        m: Tamaño de las ventanas en las que aplicar el filtro
        Q: Parametro de la media contraharmonica
    Returns:
        img_f: Imagen filtrada

    """
    img_windows = windowize(image,m) #--> split in windows
    top = np.sum(np.power(img_windows.astype('float32')+0.0000001, Q+1), 
                 axis=(2,3))
    bottom = np.sum(np.power(img_windows.astype('float32')+0.0000001, Q), 
                    axis=(2,3))
    img_f = top/bottom
    if m%2==0:
        img_f=img_f[0:-1,0:-1]
    return img_f.astype("uint8")


#############################################################################
# Order filters
#############################################################################
def median_filter(image, m=3):
    """Filtro de mediana

    Parameters:
        image: Imagen a la que se le va a aplicar el filtro
        m: Tamaño de las ventanas en las que aplicar el filtro
    Returns:
        img_f: Imagen filtrada

    """
    img_windows = windowize(image,m) #--> split in windows
    img_f = np.median(img_windows, axis=(2,3))
    if m%2==0:
        img_f=img_f[0:-1,0:-1]
    return img_f.astype("uint8")


def midpoint_filter(image, m=3):
    """Filtro de mediana

    Parameters:
        image: Imagen a la que se le va a aplicar el filtro
        m: Tamaño de las ventanas en las que aplicar el filtro
    Returns:
        img_f: Imagen filtrada

    """
    img_windows = windowize(image,m) #--> split in windows
    img_f=(np.max(img_windows.astype('float32'), axis=(2,3)) +
           np.min(img_windows.astype('float32'), axis=(2,3)))/2
    if m%2==0:
        img_f=img_f[0:-1,0:-1]
    return img_f.astype("uint8")


def alpha_trimmed_mean_filter(image, m=3, d=2):
    """Filtro de media alpha-trimmed

    Parameters:
        image: Imagen a la que se le va a aplicar el filtro
        m: Tamaño de las ventanas en las que aplicar el filtro
        d: Parametro del filtro
    Returns:
        img_f: Imagen filtrada

    """
    img_windows = windowize(image,m) #--> split in windows
    proportion = d/(m*m*2)
    w_row,w_col,_,_ = img_windows.shape
    img_windows = img_windows.reshape((w_row,w_col,m*m))
    img_f = trim_mean(img_windows,proportion,axis=2)
    if m%2==0:
        img_f=img_f[0:-1,0:-1]
    return img_f.astype("uint8")


#############################################################################
# Adaptive filters
#############################################################################
def adaptative_local_filter(image, m=3, vn=5):
    """Filtro adaptativo local

    Parameters:
        image: Imagen a la que se le va a aplicar el filtro
        m: Tamaño de las ventanas en las que aplicar el filtro
        vn: Parametro del filtro
    Returns:
        img_f: Imagen filtrada

    """
    img_windows = windowize(image, m) #--> split in windows
    # Calculate mean and variance of every window
    W_mean = np.mean(img_windows.astype('float32'), axis=(2,3))
    W_var = np.var(img_windows.astype('float32'), axis=(2,3))+0.000000001 
    # Fix inconsistencies
    aux = vn > W_var
    W_var[aux] = vn
    if m%2==0: 
        W_mean=W_mean[0:-1,0:-1] 
        W_var=W_var[0:-1,0:-1]

    img_f = image.astype("float32") - vn/W_var * (image - W_mean)
    img_f = cv2.normalize(img_f,None,0,255,cv2.NORM_MINMAX) 
    return img_f.astype("uint8")

