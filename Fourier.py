import numpy as np
import matplotlib.pylab as plt
import scipy.io.wavfile as wav 
from scipy.fftpack import fft, fftfreq,ifft
from scipy.interpolate import interp1d

#_________Punto 1____________
datos= np.genfromtxt('signal.dat',  delimiter=',')
incompletos=np.genfromtxt('incompletos.dat', delimiter=',')

#_________Punto 2____________
x= datos[:,0]
y= datos[:,1]
plt.figure()
plt.plot(x,y)
plt.title('Signal')
plt.savefig('MartinezAndrea_signal.pdf')
#_________Punto 3_____________

n = np.shape(datos)
dt = (datos[1,0]-datos[0,0])
FNy = 1/(2*dt)
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

#_________Punto 4____________

freq1 = fftfreq(n[0],dt)
mag = abs(coef)
plt.figure()
plt.plot(freq1,mag)
plt.title('Transformada de Fourier')
plt.xlabel('Frecuencias(Hz)')
plt.ylabel('Amplitud')
plt.savefig('MartinezAndrea_TF.pdf')

#__________Punto 5____________
print "Las frecuencicias principales son:", freq1[4],freq1[6], freq1[11]

# ___________Punto 6___________
def filtro (coef,freq1):
	N = len(freq1)
	for i in range (N):
		if (freq1[i] > 1000):
			coef[i]=0	
		if (freq1[i] < -1000):
			coef[i]=0			
	return coef
fil= filtro(coef,freq1)

def inversa(fil):
	inversa=ifft(np.array(fil))
	return inversa.real

t_inversa= inversa(fil)
plt.figure()
plt.plot(t_inversa,datos[:,0])
plt.show()

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
