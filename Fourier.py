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
def DFT(data):
    #xn = datos[:,1]
    xn=data
    n = np.shape(data)
    pi = np.pi
    dft = np.zeros((n[0],1), dtype = complex)
    for i in range (n[0]):
        xk = np.zeros((n[0],1), dtype = complex)
        for k in range (n[0]):
            xk[k] = xn[k]*(np.exp((-2j*pi*i*k)/(n[0])))
        dft[i] = (sum(xk))
    return dft
coef = DFT(datos[:,1])

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
plt.plot(datos[:,0],t_inversa)
plt.savefig('MartinezAndrea_filtrada.pdf')
#_____________Punto 7____________



#_______________Punto8____________
#Interpolacion cuadratica u cubica de incompletos- tranformada de Fourier de datos incompletos
x1= incompletos[:,0]
y1=incompletos[:,1]

xmin=min(x1)
xmax=max(x1)
ymin=min(y1)
ymax=max(y1)

x=np.linspace(xmin,xmax,512)




cuadratica= interp1d(x1,y1,kind='quadratic')
cubica= interp1d(x1,y1,kind='cubic')
	
int_cua=cuadratica(x)
int_cu=cubica(x)

coefcua = DFT(int_cua)
coefcu=DFT(int_cu)

n1=np.shape(int_cua)
n2=np.shape(int_cu)

#_________Punto 9____________
dt1 = (x1[1]-x1[0])
mag1=abs(coefcua)
mag2=abs(coefcu)

freq2= fftfreq(len(x),dt1)


plt.figure()
plt.subplot(3,1,1)
plt.title('Transformada de Fourier senal original')
plt.plot(freq1,mag)
plt.subplot(3,1,2)
plt.title('Transformada de Fouerie interpolacion cuadratica')
plt.plot(freq2,mag1)
plt.subplot(3,1,3)
plt.plot(freq2,mag2)
plt.title('Transformada de Fourier interpolacion cubica')
plt.xlabel('Frecuencias(Hz)')
plt.ylabel('Amplitud')
plt.savefig('MartinezAndrea_TF_interpol.pdf')
#___________Punto 10_____________

#__________Punto 11________________


def filtro2 (coef,freq1):
	N = len(freq1)
	for i in range (N):
		if (freq1[i] > 1000):
			coef[i]=0	
		if (freq1[i] < -1000):
			coef[i]=0
		if (freq1[i] < 500):
			coef[i]=0
		if(freq1[i]< -500):
			coef[i]=0			
	return coef
fil2= filtro2(coef,freq1)
fil2= filtro2(coefcua,freq2)
fil2= filtro2(coefcua,freq2)




