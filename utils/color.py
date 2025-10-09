import cv2
import numpy as np

def color_slicing_sphere(img, a, R0, bg_color=[0, 0, 0]):
    """
    Funcion para color slicing en una esfera de radio R0 centrada en el color 
    central "a".

    Parameters:
        img: Imagen BGR 
        a: Color central en [B,G,R] 
        R0: radio de la esfera
        bg_color: Color de fondo (por defecto [0, 0, 0])
    
    Returns:
        img_sliced: Imagen con el color slicing
        mask: Mascara binaria con los pixeles que estan dentro de la esfera
    """
    img = img.astype(np.float32)
    dist = np.sum((img - a)**2, axis=2)
    mask = dist < (R0**2)
    img_sliced = img.copy()
    img_sliced[~mask] = bg_color
    img_sliced = img_sliced.astype(np.uint8)
    return img_sliced, mask


def color_slicing(img, lower, upper, bg_color=[0, 0, 0]):
    """
    Funcion para color slicing en una imagen.
    Recibe la imagen y dos arrays con los valores minimo y maximo de cada 
    canal. Devuelve una imagen en el mismo espacio de color que la imagen de 
    entrada (aplica tanto a BGR como a HSV).

    Parameters:
        img: Imagen BGR o HSV
        lower: Array con el valor minimo de cada canal
        upper: Array con el valor maximo de cada canal
        bg_color: Color de fondo (por defecto [0, 0, 0])
    
    Returns:
        img_sliced: Imagen con el color slicing
        mask: Mascara binaria con los pixeles que estan dentro de la esfera
    """
    lower = np.array(lower, dtype=img.dtype)
    upper = np.array(upper, dtype=img.dtype)
    mask = cv2.inRange(img, lower, upper).astype(bool)
    img_sliced = img.copy()
    img_sliced[~mask] = bg_color
    img_sliced = img_sliced.astype(np.uint8)
    return img_sliced, mask


def complementary_color(img):
    """
    Funcion para obtener el color complementario de una imagen.
    Recibe y devuelve una imagen en BGR (se convierte a HSV internamente).

    Parameters:
        img: Imagen BGR 
    
    Returns:
        img_complement: Imagen con el color complementario
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue = img_hsv[:,:,0]
    hue += 90
    hue[hue > 179] = hue[hue > 179] - 179
    img_complement = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    return img_complement


def color_slicing_hsv(img, lower, upper, bg_color=[0, 0, 0]):
    """
    Funcion para rebanado de color (color slicing) en una imagen en el espacio de color HSV.
    Devuelve una imagen en el mismo espacio de color que la imagen de entrada.

    Parameters:
        img: Imagen BGR
        lower: Array con el valor minimo de cada canal
        upper: Array con el valor maximo de cada canal
        bg_color: Color de fondo (por defecto [0, 0, 0])
    
    Returns:
        img_sliced: Imagen con el color slicing
        mask: Mascara binaria con los pixeles que estan dentro del cubo HSV
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array(lower, dtype=img.dtype)
    upper = np.array(upper, dtype=img.dtype)
    mask = cv2.inRange(img_hsv, lower, upper).astype(bool)
    img_sliced = img.copy()
    img_sliced[~mask] = bg_color
    img_sliced = img_sliced.astype(np.uint8)
    return img_sliced, mask

def RGB2HSI(img):
    """
    Funcion para convertir una imagen en RGB a HSI.
    Devuelve una imagen en el mismo espacio de color que la imagen de entrada.
    """
    img_hsv = cv2.cvtColor(img[:, :, ::-1], cv2.COLOR_BGR2HSV)
    R, G, B = cv2.split(img.astype(np.float32))
    I = (R + G + B) / 3
    I = I.astype(np.uint8)
    img_hsi = cv2.merge((img_hsv[:,:,0], img_hsv[:,:,1], I))

    # Estas formulas no funcionan
    # Gonzales y Woods dan las formulas pensando en RGB como [0, 1]
    #R, G, B = cv2.split(img.astype(np.float32))
    #R, G, B = R/255.0, G/255.0, B/255.0

    ## H esta en [0, 360)
    #tht=np.arccos((.5*((R-G)+(R-B)))/np.sqrt((R-G)**2+((R-B)*(G-B))+0.000001))
    #tht = np.degrees(tht)
    #idx = np.where(B > G)[0]
    #H = tht.copy()
    #H[idx] = 360 - tht[idx]

    ## S y I estan en [0, 1]
    #S = 1 - 3*np.min((R, G, B), axis=0)/np.sum((R, G, B), axis=0)

    ## I es facil de calcular: promedio de los canales RGB
    #I = (R+G+B)/3.0

    ## Hasta acá, estamos en [0, 360), [0, 1], [0, 1] en forma de floats
    ## pero queremos enteros en [0, 180], [0, 255], [0, 255]
    #H, S, I = H/2, S*255, I*255
    #H, S, I = H.astype(np.uint8), S.astype(np.uint8), I.astype(np.uint8)

    #img_hsi = cv2.merge((H, S, I))
    return img_hsi


def HSI2RGB(img):
    """
    Funcion para convertir una imagen en HSI a RGB.
    Devuelve una imagen en el mismo espacio de color que la imagen de entrada.
    No funciona bien, seguramente algunas conversiones de int a float o al 
    reves se estan haciendo mal
    """
    H, S, I = cv2.split(img.astype(np.float32))

    # Recordar que H esta en [0, 180), lo convertimos a [0, 360)
    # S e I estan en [0, 255], lo convertimos a [0, 1]
    H, S, I = H*2, S/255.0, I/255.0
    
    R, G, B = np.zeros_like(H), np.zeros_like(H), np.zeros_like(H)

    # Gonzales y Woods separan el calculo de las componentes RGB por sectores
    RGi = np.where(H<120)[0]
    GBi = np.where((H>=120) & (H<240))[0]
    BRi = np.where(H>=240)[0]

    # Sector RG
    B[RGi] = I[RGi]*(1-S[RGi])
    R[RGi] = I[RGi]*(1+S[RGi]*np.cos(np.radians(H[RGi]))/ 
                     (np.cos(np.radians(60-H[RGi]))))
    G[RGi] = 3*I[RGi]-(R[RGi]+B[RGi])

    # Sector GB
    H[GBi] = H[GBi]-120
    R[GBi] = I[GBi]*(1-S[GBi])
    G[GBi] = I[GBi]*(1+S[GBi]*np.cos(np.radians(H[GBi]))/ 
                     (np.cos(np.radians(60-H[GBi]))))
    B[GBi] = 3*I[GBi]-(R[GBi]+G[GBi])

    # Sector BR
    H[BRi] = H[BRi]-240
    G[BRi] = I[BRi]*(1-S[BRi])
    B[BRi] = I[BRi]*(1+S[BRi]*np.cos(np.radians(H[BRi]))/ 
                     (np.cos(np.radians(60-H[BRi]))))
    R[BRi] = 3*I[BRi]-(G[BRi]+B[BRi])

    # Convertimos de [0, 1] a [0, 255]
    R, G, B = np.clip(R, 0, 255), np.clip(G, 0, 255), np.clip(B, 0, 255)
    R, G, B = R.astype(np.uint8), G.astype(np.uint8), B.astype(np.uint8)

    img_rgb = cv2.merge((R, G, B))
    return img_rgb


