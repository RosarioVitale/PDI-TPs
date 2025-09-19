import numpy as np
import cv2

KBOXLOW, KCROSSLOW, KBOXHI1, KBOXHI0, KCROSSHI1, KCROSSHI0 = 0, 1, 2, 3, 4, 5

def generateKernel(method, size=(3,3)):

    # Limitamos el tamano a numeros impares
    assert size[0]%2==1 and size[1]%2==1, "El kernel debe tener tamano impar"

    if method == KBOXLOW:
        mask = np.ones(size, np.float32)/(size[0]*size[1])
    elif method == KCROSSLOW:
        mask = np.zeros(size, np.float32)
        mask[size[0]//2, :] = 1/(size[0]+size[1]-1)
        mask[:, size[1]//2] = 1/(size[0]+size[1]-1)
    elif method == KBOXHI0:
        mask = -1*np.ones(size, np.float32)
        mask[size[0]//2, size[1]//2] = size[0]*size[1]-1
    elif method == KCROSSHI0: 
        mask = np.zeros(size, np.float32) 
        mask[size[0]//2, :] = -1 
        mask[:, size[1]//2] = -1 
        mask[size[0]//2, size[1]//2] = size[0]+size[1]-2 
    elif method == KBOXHI1: 
        mask = -1*np.ones(size, np.float32) 
        mask[size[0]//2, size[1]//2] = size[0]*size[1]
    elif method == KCROSSHI1: 
        mask = np.zeros(size, np.float32) 
        mask[size[0]//2, :] = -1 
        mask[:, size[1]//2] = -1 
        mask[size[0]//2, size[1]//2] = size[0]+size[1]-1 
    else: 
        print("Seleccione un metodo valido") 

    return mask
