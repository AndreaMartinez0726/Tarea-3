import numpy as np 
import matplotlib.pylab as plt
from numpy.linalg import *
from scipy.linalg import expm,inv

#________Punto 1______________

dat= np.genfromtxt('WDBC.dat',delimiter=',')
dati = np.genfromtxt('WDBC.dat', delimiter= ',' , dtype = None)

datos=dat[:,1:]

# Maligno =0; Beningino=0
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


for i in range (len(vectores)):
	vectorpos = vectores[i]
	print "el valor mas importante para el vector", i
	print ":", np.max(vectorpos)
	
	
#___________Punto 5___________

pc1 = vectores[0]
pc2 = vectores[1]
print pc1
print pc2
plt.figure()
plt.scatter(pc1,pc2)
plt.show()
	
#https://plot.ly/ipython-notebooks/principal-component-analysis/#3--projection-onto-the-new-feature-space
		




