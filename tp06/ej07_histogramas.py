import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
import utils.noise as ns
import utils.histogram as ht

IMAGE_DIR = "../imgs/"
IMAGE_FILE1 = "FAMILIA_a.jpg"
imagen1 = cv2.imread(f"{IMAGE_DIR}{IMAGE_FILE1}", cv2.IMREAD_GRAYSCALE)
IMAGE_FILE2 = "FAMILIA_b.jpg"
imagen2 = cv2.imread(f"{IMAGE_DIR}{IMAGE_FILE2}", cv2.IMREAD_GRAYSCALE)
IMAGE_FILE3 = "FAMILIA_c.jpg"
imagen3 = cv2.imread(f"{IMAGE_DIR}{IMAGE_FILE3}", cv2.IMREAD_GRAYSCALE)

#ROIXa, ROIXb, ROIYa, ROIYb = 1620, 1900, 450, 680
#ROIXa, ROIXb, ROIYa, ROIYb = 1610, 1910, 450, 550
ROIXa, ROIXb, ROIYa, ROIYb = 1810, 1910, 450, 550
#ROIXa, ROIXb, ROIYa, ROIYb = 1600, 1900, 180, 300

fig1, ax1 = plt.subplots(1, 2, layout="constrained")
#fig1, ax1 = plt.subplots(2, 1, layout="constrained")
#fig1, ax1 = plt.subplots(2, 1)
fig1.suptitle("Familia a", fontsize=16)
ax1[0].imshow(imagen1[ROIYa:ROIYb, ROIXa:ROIXb], cmap="gray", vmax=255, vmin=0)
ax1[0].set_title("ROI elegida")
hist1 = ht.get_histogram(imagen1[ROIYa:ROIYb, ROIXa:ROIXb])
#ax1[1].plot(hist1, "-*")
mk1,ln1,_ = ax1[1].stem(hist1, linefmt='b', basefmt='b')
plt.setp(mk1, markersize=1)
plt.setp(ln1, linewidth=2)
ax1[1].set_title("Histograma")

inds1=np.where(hist1>0)[0]
print(f"HISTOGRAMA 1: media={np.mean(inds1)}, std={np.std(inds1)}, {inds1}")

#fig2, ax2 = plt.subplots(2, 1, layout="constrained")
fig2, ax2 = plt.subplots(1, 2, layout="constrained")
fig2.suptitle("Familia b", fontsize=16)
ax2[0].imshow(imagen2[ROIYa:ROIYb, ROIXa:ROIXb], cmap="gray", vmax=255, vmin=0)
ax2[0].set_title("ROI elegida")
hist2 = ht.get_histogram(imagen2[ROIYa:ROIYb, ROIXa:ROIXb])
#ax2[1].plot(hist2, "-*")
mk2,ln2,_ = ax2[1].stem(hist2, linefmt='b', basefmt='b')
plt.setp(mk2, markersize=1)
plt.setp(ln2, linewidth=2)
ax2[1].set_title("Histograma")

inds2=np.where(hist2>0)[0]
mea2=np.mean(inds2)
med2=np.median(inds2)
min2=(inds2[-1]-inds2[0])//2
max2=(inds2[-1]-inds2[0])//2
print(f"HISTOGRAMA 2: media={mea2}, mediana={med2}, min={min2}, {inds2}")

#fig3, ax3 = plt.subplots(2, 1, layout="constrained")
fig3, ax3 = plt.subplots(1, 2, layout="constrained")
fig3.suptitle("Familia c", fontsize=16)
ax3[0].imshow(imagen3[ROIYa:ROIYb, ROIXa:ROIXb], cmap="gray", vmax=255, vmin=0)
ax3[0].set_title("ROI elegida")
hist3 = ht.get_histogram(imagen3[ROIYa:ROIYb, ROIXa:ROIXb])
#ax3[1].plot(hist3, "-*")
mk3,ln3,_ = ax3[1].stem(hist3, linefmt='b', basefmt='b')
plt.setp(mk3, markersize=1)
plt.setp(ln3, linewidth=2)
ax3[1].set_title("Histograma")

acum3 = np.sum(hist3)
print(f"HISTOGRAMA 3:")
print(f" + P_a0={(hist3[0]/acum3)[0]}\n + P_a1={(hist3[1]/acum3)[0]}")
print(f" + P_b0={(hist3[255]/acum3)[0]}\n + P_b1={(hist3[254]/acum3)[0]}")
#print(hist3[:5], hist3[-5:])

imagen = cv2.rectangle(cv2.cvtColor(imagen1, cv2.COLOR_GRAY2RGB),
                       (ROIXa,ROIYa), (ROIXb,ROIYb), (255,0,0),2)
fig4, ax4 = plt.subplots(1, 1, layout="constrained")
fig4.suptitle("ROI elegida")
ax4.imshow(imagen, cmap="gray", vmax=255, vmin=0)

plt.show()
