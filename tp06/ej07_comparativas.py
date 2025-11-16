import cv2
import matplotlib.pyplot as plt

IMAGE_FILE = "../imgs/FAMILIA_a.jpg"
imagen = cv2.imread(f"{IMAGE_FILE}", cv2.IMREAD_GRAYSCALE)
IMGM1_FILE = "capturas_familia/FAMILIA_a_cntArm_M3_Q0.jpg"
img_m1 = cv2.imread(f"{IMGM1_FILE}", cv2.IMREAD_GRAYSCALE)
met1="Contra-armonica (M=3,Q=0)"
IMGM2_FILE = "capturas_familia/FAMILIA_a_cntArm_M3_Qm1.jpg"
img_m2 = cv2.imread(f"{IMGM2_FILE}", cv2.IMREAD_GRAYSCALE)
met2="Contra-armonica (M=3,Q=-1)"

#IMAGE_FILE = "../imgs/FAMILIA_b.jpg"
#imagen = cv2.imread(f"{IMAGE_FILE}", cv2.IMREAD_GRAYSCALE)

#IMAGE_FILE3 = "../imgs/FAMILIA_c.jpg"
#imagen = cv2.imread(f"{IMAGE_FILE}", cv2.IMREAD_GRAYSCALE)


fig, ax = plt.subplots(1, 2, layout="constrained")
ax[0].imshow(img_m1, cmap="gray", vmax=255, vmin=0)
ax[0].set_title(met1)
ax[0].set_axis_off()
ax[1].imshow(img_m2, cmap="gray", vmax=255, vmin=0)
ax[1].set_title(met2)
ax[1].set_axis_off()

plt.show()
