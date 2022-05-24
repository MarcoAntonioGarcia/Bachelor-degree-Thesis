print('\033[H\033[J')
import numpy as np
from scipy.integrate import complex_ode
import qutip as qt
import numpy as np 
import math
import scipy as sp
from scipy.integrate import complex_ode
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def cj(z): return z.conjugate()

sx=np.matrix(([[0, 1],[1, 0]]),complex)
sy=np.matrix(([[0, -1j],[1j, 0]]),complex)
sz=np.matrix(([[1, 0],[0, -1]]),complex)
ide=np.matrix([[1,0],[0,1]])
r1=0.5;
r2=0;
r3=0.8;

tau=10;             #tiempo disipativo
tao=1;              #tiempo de acción de la compueta
t0=34.5625;
t1=t0-tao/2;
t2=t0+tao/2;
w=1;
hb=1;
kb=1;
tau=10;

rho0=(1/2)*(r1*sx+r2*sy+r3*sz+ide);
rho1=np.asarray(rho0).reshape(1,4);

def f(t,rho):    
    rho=rho.reshape(2,2)
    thetaB=math.exp(-((t-t0)/tao)**2);
    thetaG=(math.sqrt(math.pi)/(2*tao))*math.exp(-((t-t0)/tao)**2);
    H=(1/2)*w*(1-thetaB)*sz+hb*thetaG*sx;
    gamma=np.trace(np.dot(rho,np.dot(H,H)))-(np.trace(np.dot(rho,H)))**2;
    alpha=(np.trace(np.dot(rho,H))*np.trace(np.dot(rho,np.dot(H,sp.linalg.logm(rho))))-np.trace(np.dot(rho,np.dot(H,H)))*np.trace(np.dot(rho,sp.linalg.logm(rho))))/gamma;
    beta=(np.trace(np.dot(rho,sp.linalg.logm(rho)))*np.trace(np.dot(rho,H))-np.trace(np.dot(rho,np.dot(H,sp.linalg.logm(rho)))))/gamma;
    D=(np.dot(rho,sp.linalg.logm(rho)))+(alpha*rho)+((1/2)*beta*(np.dot(H,rho)+np.dot(rho,H))); 
    #D=0
    drho=(-(1j/hb)*(np.dot(H,rho)-np.dot(rho,H))-D/tau);
    return drho.reshape(1,4)

r = complex_ode(f)
r.set_initial_value(rho1, 0)

t = 50
dt = .1
i=0

Px=np.zeros(((int(t/dt+1), 1)),complex)
Py=np.zeros(((int(t/dt+1), 1)),complex)
Pz=np.zeros(((int(t/dt+1), 1)),complex)

while r.successful() and r.t < t:
    r.integrate(r.t+dt)
    Px[i]=r.y[2]+r.y[1]
    Py[i]=(1/(1j))*(r.y[2]-r.y[1])
    Pz[i]=(r.y[0]-r.y[3])
    i=i+1
    #print(i)
    #print(r.y)
    
b = qt.Bloch()  
b.frame_alpha=.1
b.sphere_alpha=.2
b.view =[0,30]
b.point_color=['y'] 
pnt=[Px.real,Py.real,Pz.real]
b.add_points(pnt)
b.show()


    

    
    
    
    
    
    
    
    
    
    
    