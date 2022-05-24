print('\033[H\033[J')
import numpy as np
from scipy.integrate import complex_ode
import scipy as sp
import math
from scipy . constants import hbar

sx=np.matrix(([[0, 1],[1, 0]]),complex)
sy=np.matrix(([[0, -1j],[1j, 0]]),complex)
sz=np.matrix(([[1, 0],[0, -1]]),complex)
ide=np.matrix([[1,0],[0,1]])

Had=1/np.sqrt(2)*np.matrix([[1,1],[1,-1]])
w=.2675;
#w=1
hb=1;

#tiempos de evolución de compuertas de 1 qubit


def Hadamard_inicial(r1H,r2H,r3H,tauH,tH,dtH):

   iH=0
   taoH=.5;            #tiempo de accion de la compueta
   t0H=10


   rho0H=(1/2)*(r1H*sx+r2H*sy+r3H*sz+ide);
   rho1H=np.asarray(rho0H).reshape(1,4);
   Com=Had

   def fH(t,rho):    
    
       rhoH=rho.reshape(2,2)
       thetaBH=math.exp(-((t-t0H)/taoH)**2);
       thetaGH=(math.sqrt(math.pi)/(2*taoH))*math.exp(-((t-t0H)/taoH)**2);
       H=-(1/2)*w*hb*(1-thetaBH)*sz+hb*thetaGH*Com;
       gammaH=np.trace(np.dot(rhoH,np.dot(H,H)))-(np.trace(np.dot(rhoH,H)))**2;
       alphaH=(np.trace(np.dot(rhoH,H))*np.trace(np.dot(rhoH,np.dot(H,sp.linalg.logm(rhoH))))-np.trace(np.dot(rhoH,np.dot(H,H)))*np.trace(np.dot(rhoH,sp.linalg.logm(rhoH))))/gammaH;
       betaH=(np.trace(np.dot(rhoH,sp.linalg.logm(rhoH)))*np.trace(np.dot(rhoH,H))-np.trace(np.dot(rhoH,np.dot(H,sp.linalg.logm(rhoH)))))/gammaH;
       D=(np.dot(rhoH,sp.linalg.logm(rhoH)))+(alphaH*rhoH)+((1/2)*betaH*(np.dot(H,rhoH)+np.dot(rhoH,H))); 
       #D=0
       drho=(-(1j/hb)*(np.dot(H,rhoH)-np.dot(rhoH,H))-D/tauH);
       return (drho.reshape(1,4))

   rH = complex_ode(fH)
   rH.set_initial_value(rho1H, 0)

   while rH.successful() and (iH+1+dtH)*dtH < tH:
       rH.integrate(rH.t+dtH)
       rhoH=rH.y.reshape(2,2)
       iH=iH+1

   return rhoH



######################Evolución del qubit objetivo sin compuerta#######################
#######################################################################################
def Qubit_objetivo(r1S,r2S,r3S,tauS,tS,dtS):

   iS=0
   rho0S=(1/2)*(r1S*sx+r2S*sy+r3S*sz+ide);
   rho1S=np.asarray(rho0S).reshape(1,4);

   def f(t,rho):    
       rho=rho.reshape(2,2)
       H=-(1/2)*w*hb*sz
       gamma=np.trace(np.dot(rho,np.dot(H,H)))-(np.trace(np.dot(rho,H)))**2;
       alpha=(np.trace(np.dot(rho,H))*np.trace(np.dot(rho,np.dot(H,sp.linalg.logm(rho))))-np.trace(np.dot(rho,np.dot(H,H)))*np.trace(np.dot(rho,sp.linalg.logm(rho))))/gamma;
       beta=(np.trace(np.dot(rho,sp.linalg.logm(rho)))*np.trace(np.dot(rho,H))-np.trace(np.dot(rho,np.dot(H,sp.linalg.logm(rho)))))/gamma;
       D=(np.dot(rho,sp.linalg.logm(rho)))+(alpha*rho)+((1/2)*beta*(np.dot(H,rho)+np.dot(rho,H))); 
       #D=0
       drho=(-(1j/hb)*(np.dot(H,rho)-np.dot(rho,H))-D/tauS);
       return (drho.reshape(1,4))

   rS = complex_ode(f)
   rS.set_initial_value(rho1S, 0)
 
   while rS.successful() and (iS+1+dtS)*dtS < tS:
       rS.integrate(rS.t+dtS)
       rhoS=rS.y.reshape(2,2)
       iS=iS+1
   return rhoS
   