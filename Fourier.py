import numpy as np
import matplotlib.pylab as plt
import scipy.io.wavfile as wav 
from scipy.fftpack import fft, fftfreq
from scipy.interpolate import interp1d

#Almacenar datos
datos= np.genfromtxt('signal.dat',  delimiter=',')
incompletos=np.genfromtxt('incompletos.dat', delimiter=',')

#Grafica de signal.dat
x= datos[:,0]
y= datos[:,1]
plt.figure()
plt.plot(x,y)
plt.title('Signal')
plt.savefig('MartinezAndrea_signal.pdf')
#Transformada discreta de Fourier a los datos signal

n = np.shape(datos)
def DFT():
    xn = datos[:,1]
    pi = np.pi
    dft = np.zeros((n[0],1), dtype = complex)
    for i in range (n[0]):
        xk = np.zeros((n[0],1), dtype = complex)
        for k in range (n[0]):
            xk[k] = xn[k]*(np.exp((-2j*pi*i*k)/(n[0])))
        dft[i] = (sum(xk))
    return dft
coef = DFT()


#Grafica transformada de Fourier



#Frecuencias principales

#Filtro, transformada inversa y grafica

#Mensaje porque no se puede hacer la transformada para los datos incompletos.dat


#Interpolacion cuadratica u cubica de incompletos- tranformada de Fourier de datos incompletos
x1= incompletos[:,0]
y1=incompletos[:,1]
def interpolacion(x1,y1):
	cuadratica= interp1d(x1,y1,kind='quadratic')
	cubica= interp1d(x1,y1,kind='cubic')
	return cuadratica,cubica




#grafica de la 3 transformadad (2 signal y 1 incompletos)




#Filtro con 1000Hz y 500Hz


#Grafica de los filtros.
