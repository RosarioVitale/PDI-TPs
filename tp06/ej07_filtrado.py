import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
import utils.noise as ns
from cvui import cvui

IMAGE_DIR = "../imgs/"

###############################################################################
#                          Imagen a: ruido gaussiano                          #
###############################################################################
IMAGE_FILE = "FAMILIA_a.jpg"
noised = cv2.imread(f"{IMAGE_DIR}{IMAGE_FILE}", cv2.IMREAD_GRAYSCALE)
print("Familia A: Contra-armonica M=3, Q=0")
filtered = ns.contraharmonic_mean_filter(noised, 3, 0)
name1="capturas_familia/FAMILIA_a_cntArm_M3_Q0.jpg"
cv2.imwrite(name1,filtered)
print("Familia A: Contra-armonica M=3, Q=-1")
filtered = ns.contraharmonic_mean_filter(noised, 3, -1)
name1="capturas_familia/FAMILIA_a_cntArm_M3_Qm1.jpg"
cv2.imwrite(name1,filtered)
print("Familia A: Adaptativo local M=7, V=352")
filtered = ns.adaptative_local_filter(noised, 7, 352)
name1="capturas_familia/FAMILIA_a_adapL_M7_V352.jpg"
cv2.imwrite(name1,filtered)
print("Familia A: Adaptativo local M=11, V=352")
filtered = ns.adaptative_local_filter(noised, 11, 352)
name1="capturas_familia/FAMILIA_a_adapL_M11_V352.jpg"
cv2.imwrite(name1,filtered)
print("Familia A: Bilateral")
filtered = cv2.bilateralFilter(noised, 11, 30, 250)
name1="capturas_familia/FAMILIA_a_bilat_D11_SC30_SS250.jpg"
cv2.imwrite(name1,filtered)

###############################################################################
#                          Imagen b: ruido uniforme                           #
###############################################################################
IMAGE_FILE = "FAMILIA_b.jpg"
noised = cv2.imread(f"{IMAGE_DIR}{IMAGE_FILE}", cv2.IMREAD_GRAYSCALE)
print("Familia B: Contra-armonica M=3 Q=0")
filtered = ns.contraharmonic_mean_filter(noised, 3, 0)
name="capturas_familia/FAMILIA_b_cntArm_M3_Q0.jpg"
cv2.imwrite(name,filtered)
print("Familia B: Contra-armonica M=3 Q=-1")
filtered = ns.contraharmonic_mean_filter(noised, 3, -1)
name="capturas_familia/FAMILIA_b_cntArm_M3_Qm1.jpg"
cv2.imwrite(name,filtered)
print("Familia B: Bilateral")
filtered = cv2.bilateralFilter(noised, 11, 30, 250)
name="capturas_familia/FAMILIA_b_bilat_D11_SC30_SS250.jpg"
cv2.imwrite(name,filtered)
print("Familia B: Adaptativo local M=5, V=48")
filtered = ns.adaptative_local_filter(noised, 5, 48)
name="capturas_familia/FAMILIA_b_adapL_M5_V48.jpg"
cv2.imwrite(name,filtered)
print("Familia B: Adaptativo local M=7, V=48")
filtered = ns.adaptative_local_filter(noised, 7, 48)
name="capturas_familia/FAMILIA_b_adapL_M7_V48.jpg"
cv2.imwrite(name,filtered)

###############################################################################
#                          Imagen c: ruido impulsivo                          #
###############################################################################
IMAGE_FILE = "FAMILIA_c.jpg"
noised = cv2.imread(f"{IMAGE_DIR}{IMAGE_FILE}", cv2.IMREAD_GRAYSCALE)
print("Familia C: Mediana M=3")
filtered = ns.median_filter(noised, 3)
name="capturas_familia/FAMILIA_c_median_M3.jpg"
cv2.imwrite(name,filtered)
print("Familia C: Mediana M=5")
filtered = ns.median_filter(noised, 5)
name="capturas_familia/FAMILIA_c_median_M5.jpg"
cv2.imwrite(name,filtered)
print("Familia C: Alfa-recortado M=3, D=9")
filtered = ns.alpha_trimmed_mean_filter(noised,3,9)
name="capturas_familia/FAMILIA_c_atrim_M3_D9.jpg"
cv2.imwrite(name,filtered)
print("Familia C: Alfa-recortado M=5, D=9")
filtered = ns.alpha_trimmed_mean_filter(noised,5,9)
name="capturas_familia/FAMILIA_c_atrim_M5_D9.jpg"
cv2.imwrite(name,filtered)

