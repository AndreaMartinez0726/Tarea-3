
from scipy import ndimage
import numpy as np
from scipy.fftpack import fft2, fftfreq,ifft,fft
import matplotlib.pylab as plt
from scipy import fftpack
#________Punto 1__________
imagen=ndimage.imread("arbol.png")

#________Punto 2__________

Fourier=fftpack.fft2(imagen)

fm= Fourier.real**2+Fourier.imag**2
fm= np.sqrt(fm)
fmm=np.log(fm)

plt.figure()
plt.imshow(fm)
#plt.axis('off')
plt.savefig('MartinezAndrea_FT2D.pdf')

#_________Punto 3 ________

tam=np.size(Fourier)

			
for i in range(tam,0):
    for j in range(tam,1):
        
        if (((i-10)/0.5)**2 + ((j-30)/2)**2 < 100):
           	Fourier[i,j] = Fourier[i,j]*((i-10)**2+(j-30)**2)/400
        if (((i-60)/0.5)**2 + ((j-60)/2)**2 < 100):
            Fourier[i,j] = Fourier[i,j]*((i-60)**2+(j-60)**2)/400
        if (((i-190)/0.5)**2 + ((j-200)/2)**2 < 100):
            Fourier[i,j] = Fourier[i,j]*((i-190)**2+(j-200)**2)/400
        if (((i-245)/0.5)**2 + ((j-230)/2)**2 < 100):
            Fourier[i,j] = Fourier[i,j]*((i-245)**2+(j-230)**2)/400



