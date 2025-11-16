import cv2
import matplotlib.pyplot as plt
import numpy as np
#import imutils
import sys
sys.path.append("..")
from utils import fourier as uf
from utils import utils as uts
import cvui

IMAGE_DIR = "parrafos/"
for i in range(5):
    IMAGE_FILE = f"parrafo{i}.jpg"
    imagen = cv2.imread(f"{IMAGE_DIR}{IMAGE_FILE}", cv2.IMREAD_GRAYSCALE)

    pad_width = uf.get_pad_width(imagen.shape)
    imagen = cv2.copyMakeBorder(imagen, pad_width[0][0], pad_width[0][1],
                                pad_width[1][0], pad_width[1][1],
                                cv2.BORDER_REPLICATE)
    img_blur=uf.apply_filter(imagen,uf.BUTTERWORTH_FILTER,{"R0":50, "n":2},True)
    mag_img, _, _ = uf.get_dft(img_blur)

    ISHY, ISHX = mag_img.shape
    print(f"Image size: {ISHX}x{ISHY}")

    fmag_img = np.copy(mag_img)
    fmag_img = cv2.equalizeHist(fmag_img)
    fmag_img = cv2.medianBlur(fmag_img,25)
    #fmag_img = cv2.threshold(fmag_img, 0, 255,
    #                         cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    fmag_img = cv2.threshold(fmag_img, 200, 255,
                             cv2.THRESH_OTSU)[1]

    center = (ISHX//2,ISHY//2)
    print(f"Center of circle: {center}")
    radius = np.min([ISHX,ISHY])//2
    print(f"Radius: {radius}")

    max_mean, max_perf, max_angle, max_seg = 0, [], 0, (0,0)
    for angle in range(180):
        CIRCX = int(radius*np.cos(np.deg2rad(angle)))
        CIRCY = int(radius*np.sin(np.deg2rad(angle)))
        seg_end = (center[0]+CIRCX,center[1]-CIRCY)
        segment = [center,seg_end] if angle<90 else [seg_end,center]
        perf = np.array(uts.get_segment(fmag_img, segment[0][0], segment[0][1],
                                        segment[1][0], segment[1][1]))
        if np.mean(perf)>max_mean:
            max_mean = np.mean(perf)
            max_perf = perf
            max_angle = angle
            max_seg = seg_end

    max_angle = float(max_angle)
    print(f"Mean: {max_mean} - Angle: {max_angle} - Segment end: {max_seg}")

    max_angle = 90. - max_angle
    print(f"New angle: {max_angle}")

    fmag_seg = cv2.cvtColor(fmag_img,cv2.COLOR_GRAY2RGB)
    cv2.line(fmag_seg,center,max_seg,(255,0,0),2)

    (h, w) = imagen.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, max_angle, 1.0)
    rotated = cv2.warpAffine(imagen, M, (ISHX, ISHY), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)

    fig, ax = plt.subplots(2, 3, layout="constrained")
    fig.suptitle("Orientar texto", fontsize=16)
    ax[0][0].imshow(imagen, cmap="gray", vmax=255, vmin=0)
    ax[0][0].set_title("Imagen original")
    ax[0][1].imshow(img_blur, cmap="gray", vmax=255, vmin=0)
    ax[0][1].set_title("Imagen filtrada")
    ax[0][2].imshow(mag_img, cmap="gray", vmax=255, vmin=0)
    ax[0][2].set_title("Magnitud")
    ax[1][0].imshow(fmag_img, cmap="gray", vmax=255, vmin=0)
    ax[1][0].set_title("Magnitud binarizada")
    ax[1][1].imshow(fmag_seg)
    ax[1][1].set_title("Segmento seleccionado")
    ax[1][2].imshow(rotated, cmap="gray", vmax=255, vmin=0)
    ax[1][2].set_title("Imagen rotada")

plt.show()
