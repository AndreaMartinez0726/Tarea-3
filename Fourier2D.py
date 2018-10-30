import numpy as np
import matplotlib.pylab as plt
from scipy import ndimage
from scipy import fftpack


#________Punto 1__________
imagen=ndimage.imread("arbol.png")

#________Punto 2__________

Fourier=fftpack.fft2(imagen)

fm= Fourier.real**2+Fourier.imag**2
fm= (fm)**(1./2.)
fm=np.log(fm)

plt.figure()
plt.imshow(fm)
plt.axis('off')
#plt.savefig('MartinezAndrea_FT2D.pdf')

#_________Punto 3 ________

tam=np.size(Fourier,0)
tam1=np.size(Fourier,1)

			
for i in range(tam):
    for j in range(tam1):
        
        if (((i-10)/0.5)**2 + ((j-30)/2)**2 < 100):
           	Fourier[i,j] = Fourier[i,j]*((i-10)**2+(j-30)**2)/400
        if (((i-60)/0.5)**2 + ((j-60)/2)**2 < 100):
            Fourier[i,j] = Fourier[i,j]*((i-60)**2+(j-60)**2)/400
        if (((i-190)/0.5)**2 + ((j-200)/2)**2 < 100):
            Fourier[i,j] = Fourier[i,j]*((i-190)**2+(j-200)**2)/400
        if (((i-245)/0.5)**2 + ((j-230)/2)**2 < 100):
            Fourier[i,j] = Fourier[i,j]*((i-245)**2+(j-230)**2)/400


fm= Fourier.real**2+Fourier.imag**2
fm= (fm)**(1./2.)
fm=np.log(fm)

#_______Punto 4_________
plt.figure()
plt.imshow(fm)
plt.axis('off')
#plt.savefig('MartinezAndrea_FT2D_filtrada.pdf')

#_______Punto 5________

inversa=fftpack.ifft2(Fourier)
plt.figure()
plt.imshow(inversa.real,cmap=plt.cm.gray)
plt.axis('off')
#plt.savefig('MartinezAndrea_Imagen_filtrada.pdf')
plt.show()



