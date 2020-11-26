import cv2 as cv
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sys
img=cv.imread("E:\DIP\Assign_01\q1.jpg")
f_xy = np.array([[0, 1, 2, 1], [1, 2, 3, 2], [2, 3, 4, 3], [1, 2, 3, 2]])
T = np.array([[1, 1, 1, 1], [1, -1j, -1, 1j], [1, -1, 1, -1], [1, 1j, -1,-1j]], dtype=complex)
F_uv = T*f_xy*T
print("f(x,y)")
print(f_xy)
print("F(u,v)")
print(F_uv)
def dft_2d(img):
    data = np.asarray(img)
    M, N = img.shape
    dft2d = np.zeros((M, N), dtype=complex)
    for k in range(M):
        for l in range(N):
            sum_matrix = 0.0
            for m in range(M):
                for n in range(N):
                    e = np.exp(-2j*np.pi*((k*m)/M+(l*n)/N))
                    sum_matrix += data[m,n]*e
                dft2d[k, l] = sum_matrix
    return dft2d
# Takes in a RGB image, converts to grayscale
img_rgb = img
img = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
img_rgb = cv.cvtColor(img_rgb, cv.COLOR_BGR2RGB)
# Taking the DFT and zero centering it
img_dft = sp.fft.fftshift(dft_2d(img))
img_rgb = cv.cvtColor(img_rgb, cv.COLOR_BGR2RGB)
# Plotting the original images and the DFT Spectrum
fig = plt.figure()
ax = fig.add_subplot(221)
ax.imshow(img_rgb)
ax.set_title('Original', fontsize=10)
ax.axis('off')
ax = fig.add_subplot(222)
ax.imshow(img, cmap='gray')
ax.set_title('Grayscale', fontsize=10)
ax.axis('off')
ax = fig.add_subplot(223)
ax.imshow(np.log10(np.abs(img_dft)), cmap='gray')
ax.set_title('Magnitude Spectrum', fontsize=10)
ax.axis('off')
ax = fig.add_subplot(224)
ax.imshow(np.angle(img_dft), cmap='gray')
ax.set_title('Phase Spectrum', fontsize=10)
ax.axis('off')
plt.tight_layout()
plt.show()
