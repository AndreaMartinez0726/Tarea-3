import numpy as np 
import matplotlib.pylab as plt
from numpy.linalg import *
from scipy.linalg import expm,inv

#________Punto 1______________

dat= np.genfromtxt('WDBC.dat',delimiter=',')
dati = np.genfromtxt('WDBC.dat', delimiter= ',' , dtype = None)

datos=dat[:,1:]

# Maligno =0; Benigno=1
for i in range(0,len(datos)):
	if(dati[i][1]=='M'):
		datos[i][0]=1	
	else:
		datos[i][0]=0


#__________Punto 2____________
def covarianza():
	n=np.shape(datos)[0]
	cov=np.zeros((np.shape(datos)[1],np.shape(datos)[1]))
	for i  in range (np.shape(datos)[1]):
		for j in range (np.shape(datos)[1]):
			datos[:,i]= datos[:,i] - np.mean(datos[:,i])
			datos[:,j]= datos[:,j] - np.mean(datos[:,j])
			cov[i,j]= np.sum((datos[:,i]*datos[:,j])/(n-1) )
	return cov
x= covarianza()
print 'Matriz de covarianza',x


#__________Punto 3 ___________

valores,vectores=np.linalg.eig(x)
for i in range (len(valores)):
	contador=1
	print "Autovalor", valores[i],"Correspondiente al siguiente autovector",vectores[i]
#_________Punto 4____________

#Imprima un mensaje que diga cuales son los parametros mas importantes en base a las componentes de los autovectores
#for i in range (len(vectores)):
#	vectorpos = vectores[i]
#	print "el valor mas importante para el vector", i
#	print ":", np.max(vectorpos)
	
	
#___________Punto 5___________




principales=[]
principales.append(vectores[0])
principales.append(vectores[1])
prin=np.asarray(principales)

nuevosdatos=np.dot(prin,datos.T)
plt.figure()
plt.title('Proyeccion PC1 y PC2')

plt.scatter(nuevosdatos[0,:],nuevosdatos[1,:])
plt.show()

		




