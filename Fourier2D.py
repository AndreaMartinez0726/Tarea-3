from PIL import Image
from scipy import ndimage
import numpy as np
from scipy.fftpack import fft2, fftfreq,ifft,fft
import matplotlib.pylab as plt


img=Image.open("arbol.png")


Fou=fft2(imagen)
plt.figure()
plt.imshow(Fou)
plt.show()
