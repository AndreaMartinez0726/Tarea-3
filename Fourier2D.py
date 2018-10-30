from PIL import Image
from scipy import ndimage
import numpy as np
from scipy.fftpack import fft2, fftfreq,ifft,fft
import matplotlib.pylab as plt

#________Punto 1__________
img=Image.open("arbol.png")

#________Punto 2__________
Fou=fft2(imagen)
plt.figure()
plt.imshow(Fou)
plt.show()

#_________Punto 3 ________




#_________Punto 4_________




#_________Punto 5_________
