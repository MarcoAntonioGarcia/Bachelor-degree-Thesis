print('\033[H\033[J')
import numpy as np
from scipy.integrate import complex_ode
import qutip as qt 
import scipy as sp
import matplotlib.pyplot as plt
from scipy . constants import hbar,k
from partial_trace import partial_traceA 
from partial_trace import partial_traceB
from matplotlib import cm
import matplotlib as mpl
import math

sx=np.matrix(([[0, 1],[1, 0]]),complex)
sy=np.matrix(([[0, -1j],[1j, 0]]),complex)
sz=np.matrix(([[1, 0],[0, -1]]),complex)
ide=np.matrix([[1,0],[0,1]])

Had=1/np.sqrt(2)*np.matrix([[1,1],[1,-1]])
#polarización para qubit A (Control) Esto define el estado inicial
r1H=0.001;
r2H=0;
r3H=.999;

#polarización para qubit B (Objetivo) Esto define el estado inicial
r1=0.001;
r2=0;
r3=.999;

#tiempos de evolución de compuertas de 1 qubit
t = 40e-9
dt = .1e-9
i=0

tH = t
dtH = dt
iH=i

tS = t
dtS = dt
iS=i

######################Evolución de compuerta Hadamard#######################
########################################################################

tauH=10e-9;            #tiempo disipativo compuerta Hadamard (control) y qubit (objetivo)
taoH=.5e-9;            #tiempo de accion de la compueta
t0H=33.5625e-9;
t0H=5e-9
t1H=t0H-taoH/2;
t2=t0H+taoH/2;
w=.2675e9;
hb=hbar;
kb=k;

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

PxH=np.zeros(((int(t/dt), 1)),complex)
PyH=np.zeros(((int(t/dt), 1)),complex)
PzH=np.zeros(((int(t/dt), 1)),complex)
time=np.zeros(int(t/dt),float)
dsdtSEAH=np.zeros(int(t/dt),float)
dsdtQMH=np.zeros(int(t/dt),float)
sSEAH=np.zeros(int(t/dt),float)
eigenva1H=np.zeros(int(t/dt),float)
eigenva2H=np.zeros(int(t/dt),float)
dsdtH=np.zeros(int(t/dt),float)
EH=np.zeros(int(t/dt),float)
PurityH=np.zeros(int(t/dt),float) 

while rH.successful() and (iH+1+dtH)*dtH < tH:
    rH.integrate(rH.t+dtH)
    rhoH=rH.y.reshape(2,2)
    drho1H=fH(i*dtH,rhoH)
    drhoH=drho1H.reshape(2,2)
        
    thetaBH=math.exp(-((t-t0H)/taoH)**2);
    thetaGH=(math.sqrt(math.pi)/(2*taoH))*math.exp(-((t-t0H)/taoH)**2);
    
    HH=-(1/2)*w*hb*(1-thetaBH)*sz+hb*thetaGH*Com;
    gammaH=np.trace(np.dot(rhoH,np.dot(HH,HH)))-(np.trace(np.dot(rhoH,HH)))**2;
    alphaH=(np.trace(np.dot(rhoH,HH))*np.trace(np.dot(rhoH,np.dot(HH,sp.linalg.logm(rhoH))))-np.trace(np.dot(rhoH,np.dot(HH,HH)))*np.trace(np.dot(rhoH,sp.linalg.logm(rhoH))))/gammaH;
    betaH=(np.trace(np.dot(rhoH,sp.linalg.logm(rhoH)))*np.trace(np.dot(rhoH,HH))-np.trace(np.dot(rhoH,np.dot(HH,sp.linalg.logm(rhoH)))))/gammaH;
    DH=(np.dot(rhoH,sp.linalg.logm(rhoH)))+(alphaH*rhoH)+((1/2)*betaH*(np.dot(HH,rhoH)+np.dot(rhoH,HH))); 
    drhoH=(-(1j/hb)*(np.dot(HH,rhoH)-np.dot(rhoH,HH))-DH/tauH);
    DdH=((np.dot(sp.linalg.logm(rhoH),rhoH))+(alphaH*rhoH)+((1/2)*betaH*(np.dot(rhoH,HH)+np.dot(HH,rhoH))));
    dsdtSEAH[iH]=((1/tauH)*np.trace(np.dot(DH,DdH)))/w
    dsdtSEAH[iH]=-(np.trace(np.dot(drhoH,sp.linalg.logm(rhoH)))-np.trace(drhoH))/w
    sSEAH[iH]=-np.trace(np.dot(rhoH,(1/math.log(2))*sp.linalg.logm(rhoH)))
    PurityH[iH]=np.trace(np.dot(rhoH,rhoH)) 
     
    eigH= np.linalg.eig(rhoH)
    eig2H=eigH[0]
    eigenva1H[iH]=eig2H[0]
    eigenva2H[iH]=eig2H[1]
    EH[iH]=np.trace(np.dot(rhoH,HH))  
    
    PxH[iH]=rhoH[0,1]+rhoH[1,0]
    PyH[iH]=(-1/(1j))*(rhoH[0,1]-rhoH[1,0])
    PzH[iH]=(rhoH[0,0]-rhoH[1,1])
    time[iH]=dtH*i

    iH=iH+1

######################Evolución del qubit objetivo sin compuerta#######################
#######################################################################################

rho0S=(1/2)*(r1*sx+r2*sy+r3*sz+ide);
rho1S=np.asarray(rho0S).reshape(1,4);

def f(t,rho):    
    rho=rho.reshape(2,2)
    H=-(1/2)*w*hb*sz
    gamma=np.trace(np.dot(rho,np.dot(H,H)))-(np.trace(np.dot(rho,H)))**2;
    alpha=(np.trace(np.dot(rho,H))*np.trace(np.dot(rho,np.dot(H,sp.linalg.logm(rho))))-np.trace(np.dot(rho,np.dot(H,H)))*np.trace(np.dot(rho,sp.linalg.logm(rho))))/gamma;
    beta=(np.trace(np.dot(rho,sp.linalg.logm(rho)))*np.trace(np.dot(rho,H))-np.trace(np.dot(rho,np.dot(H,sp.linalg.logm(rho)))))/gamma;
    D=(np.dot(rho,sp.linalg.logm(rho)))+(alpha*rho)+((1/2)*beta*(np.dot(H,rho)+np.dot(rho,H))); 
    #D=0
    drho=(-(1j/hb)*(np.dot(H,rho)-np.dot(rho,H))-D/tauH);
    return (drho.reshape(1,4))

rS = complex_ode(f)
rS.set_initial_value(rho1S, 0)

PxS=np.zeros(((int(t/dt), 1)),complex)
PyS=np.zeros(((int(t/dt), 1)),complex)
PzS=np.zeros(((int(t/dt), 1)),complex)
dsdtSEAS=np.zeros(int(t/dt),float)
dsdtQMS=np.zeros(int(t/dt),float)
sSEAS=np.zeros(int(t/dt),float)
eigenva1S=np.zeros(int(t/dt),float)
eigenva2S=np.zeros(int(t/dt),float)
dsdtS=np.zeros(int(t/dt),float)
ES=np.zeros(int(t/dt),float)  
PurityS=np.zeros(int(t/dt),float)  

 
while rS.successful() and (iS+1+dtS)*dtS < tS:
    rS.integrate(rS.t+dt)
    rhoS=rS.y.reshape(2,2)
    drho1=f(i*dt,rhoS)
    drho=drho1.reshape(2,2)
        
    HS=-(1/2)*w*hb*sz
    gammaS=np.trace(np.dot(rhoS,np.dot(HS,HS)))-(np.trace(np.dot(rhoS,HS)))**2;
    alphaS=(np.trace(np.dot(rhoS,HS))*np.trace(np.dot(rhoS,np.dot(HS,sp.linalg.logm(rhoS))))-np.trace(np.dot(rhoS,np.dot(HS,HS)))*np.trace(np.dot(rhoS,sp.linalg.logm(rhoS))))/gammaS;
    betaS=(np.trace(np.dot(rhoS,sp.linalg.logm(rhoS)))*np.trace(np.dot(rhoS,HS))-np.trace(np.dot(rhoS,np.dot(HS,sp.linalg.logm(rhoS)))))/gammaS;
    DS=((np.dot(rhoS,sp.linalg.logm(rhoS)))+(alphaS*rhoS)+((1/2)*betaS*(np.dot(HS,rhoS)+np.dot(rhoS,HS))));
    DdS=((np.dot(sp.linalg.logm(rhoS),rhoS))+(alphaS*rhoS)+((1/2)*betaS*(np.dot(rhoS,HS)+np.dot(HS,rhoS))));
    dsdtSEAS[iS]=((1/tauH)*np.trace(np.dot(DS,DdS)))/w
    dsdtSEAS[iS]=-(np.trace(np.dot(drho,sp.linalg.logm(rhoS)))-np.trace(drho))/w
    sSEAS[iS]=-np.trace(np.dot(rhoS,(1/math.log(2))*sp.linalg.logm(rhoS)))
    PurityS[iS]=np.trace(np.dot(rhoS,rhoS)) 
        
    eigS=np.linalg.eig(rhoS)
    eig2S=eigS[0]
    eigenva1S[iS]=eig2S[0]
    eigenva2S[iS]=eig2S[1]
    ES[iS]=np.trace(np.dot(rhoS,HS))
    
    PxS[iS]=rhoS[0,1]+rhoS[1,0]
    PyS[iS]=(-1/(1j))*(rhoS[0,1]-rhoS[1,0])
    PzS[iS]=(rhoS[0,0]-rhoS[1,1])
    time[iS]=dtS*iS
    iS=iS+1

tlistHS = np.linspace(0,tH,tH/dtH)   












######################Evolución de compuerta CNOT#######################
########################################################################
#bases para espacio de Hilbert dimencion 4

sx2=np.kron(sx,sx)     
sy2=np.kron(sy,sy)
sz2=np.kron(sz,sz)

tauA=10e-9;             #tiempo disipativo A
tauB=10e-9;             #tiempo disipativo B

#parametros de la compuerta
tao=5e-9;               #tiempo de accion de la compueta
#t0=33.5625e-9;
t0=50e-9;               #tiempo se resetea....
t1=t0-tao/2;            #tiempo de inicio de activacion de compuerta
t2=t0+tao/2;            #tiempo de termino de activacion de compuerta
w=.2675e9;              #frecuencia angular

#constantes fisicas utilizadas
hb=hbar;                #constante modificada de Planck
kb=k;                   #constante de Boltzman


FA=0.0002*hb*w          #fuerza de acoplamiento entre subsistemas
V=-FA*(sx2+sy2+sz2)     

#Hamiltonianos de subsistemas
HA=-(1/2)*(hb*w)*sz
HB=-(1/2)*(hb*w)*sz

#Estados iniciales de los qubits tomados de los resultados anteriores
rhoBi=rhoS
rhoAi=rhoH

#operador de densidad  global inicial (sin coorelacion)
rhoi=np.kron(rhoAi,rhoBi)
rho1=np.asarray(rhoi).reshape(1,16)

#operadores de densidad locales iniciales (TR_A mapea de rho-->rhoA)
rhoA=partial_traceA(rhoi,np)
rhoB=partial_traceB(rhoi,np)

#compuerta Controlled Not
C_NOT=np.matrix(([[1, 0, 0, 0 ],[0,1,0,0],[0,0,0,1],[0,0,1,0]]),complex)
H1=np.kron(HA,ide)+np.kron(ide,HB)         #ya tiene los terminos hb*w en los hamiltonianos estandar

def f(t,rho):   
    rho=rho.reshape(4,4)
    rhoA=partial_traceA(rho,np)
    rhoB=partial_traceB(rho,np)
    
    #logaritmo de rho
    lnrho=sp.linalg.logm(rho)
    lnrhoA=partial_traceA((np.dot(np.kron(ide,rhoB),lnrho)),np)
    lnrhoB=partial_traceB((np.dot(np.kron(rhoA,ide),lnrho)),np)
    
    
    thetaB=math.exp(-((t-t0)/tao)**2);
    thetaG=(math.sqrt(math.pi)/(2*tao))*math.exp(-((t-t0)/tao)**2);
    H=(1-thetaB)*H1+hb*thetaG*C_NOT+V       #si quitamos V tambien funciona
    
    
    HA=partial_traceA((np.dot(np.kron(ide,rhoB) ,H)),np)
    HB=partial_traceB((np.dot(np.kron(rhoA,ide) ,H)),np)
    
    #HA=(1/2)*(hb*w)*sz
    gammaA=np.trace(np.dot(rhoA,np.dot(HA,HA)))-(np.trace(np.dot(rhoA,HA)))**2;
    alphaA=(np.trace(np.dot(rhoA,HA))*(1/2)*np.trace(np.dot(rhoA,(np.dot(HA,lnrhoA)+np.dot(lnrhoA,HA))))-np.trace(np.dot(rhoA,np.dot(HA,HA)))*np.trace(np.dot(rhoA,lnrhoA)))/gammaA;
    betaA=(np.trace(np.dot(rhoA,lnrhoA))*np.trace(np.dot(rhoA,HA))-(1/2)*np.trace(np.dot(rhoA,(np.dot(HA,lnrhoA)+np.dot(lnrhoA,HA)))))/gammaA;
    DA=(1/2)*((np.dot(rhoA,lnrhoA))+(np.dot(lnrhoA,rhoA)))+(alphaA*rhoA)+((1/2)*betaA*(np.dot(rhoA,HA)+np.dot(HA,rhoA)));
    #DA=(1/2)*((np.dot(rhoA,sp.linalg.logm(rhoA))))+(alphaA*rhoA)+((1/2)*betaA*(np.dot(rhoA,HA)+np.dot(HA,rhoA)));
    
    #HB=(1/2)*(hb*w)*sz
    gammaB=np.trace(np.dot(rhoB,np.dot(HB,HB)))-(np.trace(np.dot(rhoB,HB)))**2;
    alphaB=(np.trace(np.dot(rhoB,HB))*(1/2)*np.trace(np.dot(rhoB,(np.dot(HB,lnrhoB)+np.dot(lnrhoB,HB))))-np.trace(np.dot(rhoB,np.dot(HB,HB)))*np.trace(np.dot(rhoB,lnrhoB)))/gammaB;
    betaB=(np.trace(np.dot(rhoB,lnrhoB))*np.trace(np.dot(rhoB,HB))-(1/2)*np.trace(np.dot(rhoB,(np.dot(HB,lnrhoB)+np.dot(lnrhoB,HB)))))/gammaB;
    DB=(1/2)*((np.dot(rhoB,lnrhoB))+(np.dot(lnrhoB,rhoB)))+(alphaB*rhoB)+((1/2)*betaB*(np.dot(rhoB,HB)+np.dot(HB,rhoB))); 
    #DB=(1/2)*((np.dot(rhoB,sp.linalg.logm(rhoB))))+(alphaB*rhoB)+((1/2)*betaB*(np.dot(rhoB,HB)+np.dot(HB,rhoB))); 
    D=((1/tauA)*(np.kron(DA,rhoB)))+((1/tauB)*(np.kron(rhoA,DB)))
    #D=0
    drho=-(1j/hb)*(np.dot(H,rho)-np.dot(rho,H))-D;
    
    return drho.reshape(1,16)

r = complex_ode(f)
r.set_initial_value(rho1, 0e-9)


t = 180e-9
dt = .1e-9
i=400

Px=np.zeros(((int(t/dt), 1)),complex)
Py=np.zeros(((int(t/dt), 1)),complex)
Pz=np.zeros(((int(t/dt), 1)),float)

Px2=np.zeros(((int(t/dt), 1)),complex)
Py2=np.zeros(((int(t/dt), 1)),complex)
Pz2=np.zeros(((int(t/dt), 1)),float)
time=np.zeros(int(t/dt),float)
Fac_Cor=np.zeros(((int(t/dt), 1)),complex)

Fac_Cor2=np.zeros(((int(t/dt), 1)),complex)

sSEA=np.zeros(int(t/dt),float)
sSEA_A=np.zeros(int(t/dt),float)
sSEA_B=np.zeros(int(t/dt),float)

Purity_A=np.zeros(int(t/dt),float)
Purity_B=np.zeros(int(t/dt),float)
dsdtSEA=np.zeros(int(t/dt),float)
sigmaAB=np.zeros(int(t/dt),float)
#E=np.zeros(((int(t/dt))),float)
Concurrencia=np.zeros(((int(t/dt))),float)
drho1=np.zeros(((int(t/dt),16)),float)
rho=np.zeros(((int(t/dt),4,4)),float)
#DrhoDt=np.zeros(((n, 16)),complex)


while r.successful() and (i+dt)*dt <= t:
    r.integrate(r.t+dt)
    rho=r.y.reshape(4,4)

    rhoA=partial_traceA(rho,np)
    rhoB=partial_traceB(rho,np)
    
    thetaB=math.exp(-((t-t0)/tao)**2);
    thetaG=(math.sqrt(math.pi)/(2*tao))*math.exp(-((t-t0)/tao)**2);
    H=(1-thetaB)*H1+hb*thetaG*C_NOT+V      #si quitamos V tambien funciona
    
    HA=partial_traceA((np.dot(np.kron(ide,rhoB) ,H)),np)
    HB=partial_traceB((np.dot(np.kron(rhoA,ide) ,H)),np)
        
    lnrho=sp.linalg.logm(rho)
    lnrhoA=partial_traceA((np.dot(np.kron(ide,rhoB),lnrho)),np)
    lnrhoB=partial_traceB((np.dot(np.kron(rhoA,ide),lnrho)),np)
            
    drho1=f(i*dt,rho)
    drho=drho1.reshape(4,4)
    
    Fac_Cor[i]=np.trace(((np.dot(H1,np.kron(rhoA,rhoB))-np.dot(np.kron(rhoA,rhoB),H1)))*(np.dot(np.kron(rhoA,rhoB),H1)-(np.dot(H1,np.kron(rhoA,rhoB)))))/((w**2)*(hb**2))
    sSEA[i]=-np.trace(np.dot(rho,sp.linalg.logm(rho)))      #entropia del sistema 
    sSEA_A[i]=-np.trace(np.dot(rhoA,sp.linalg.logm(rhoA)))  #entropia del subsistema A
    sSEA_B[i]=-np.trace(np.dot(rhoB,sp.linalg.logm(rhoB)))  #entropia del subsistema B
    sigmaAB[i]=np.trace(np.dot(rho,sp.linalg.logm(rho)))-np.trace(np.dot(rhoA,sp.linalg.logm(rhoA)))-np.trace(np.dot(rhoB,sp.linalg.logm(rhoB)))
    Purity_A[i]=np.trace(np.dot(rhoA,rhoA))                 #pureza del subsistema A
    Purity_B[i]=np.trace(np.dot(rhoB,rhoB))                 #pureza del subsistema B 
    
    #concurrencia
    rho_concurrencia=np.dot(np.dot(sy2,np.conjugate(rho)),sy2)
    rho_raiz=sp.linalg.sqrtm(rho)
    R=sp.linalg.sqrtm(np.dot(np.dot(rho_raiz,rho_concurrencia),rho_raiz))
    R_lamda=sorted(np.linalg.eigh(R)[0])
    Concurrencia[i]=R_lamda[3]-R_lamda[2]-R_lamda[1]-R_lamda[0]
    
    
    dsdtSEA[i]=-(np.trace(np.dot(drho,sp.linalg.logm(rho)))-np.trace(drho))/w
    
    time[i]=dt*i
    Px[i]=rhoA[0,1]+rhoA[1,0]
    Py[i]=(-1/(1j))*(rhoA[0,1]-rhoA[1,0])
    Pz[i]=(rhoA[0,0]-rhoA[1,1])
    
    Px2[i]=rhoB[0,1]+rhoB[1,0]
    Py2[i]=(-1/(1j))*(rhoB[0,1]-rhoB[1,0])
    Pz2[i]=(rhoB[0,0]-rhoB[1,1])
    #Fac_Cor[0]=np.trace((1j*(np.dot(H,rhoi)-np.dot(rhoi,H)))*(-1j*(np.dot(rhoi,H)-(np.dot(H,rhoi))))) 
    
    i=i+1
        
    
tlist = np.linspace(0,t,t/dt)    
    
b = qt.Bloch()  
nrm = mpl.colors.Normalize(0,t)
colors = cm.jet(nrm(tlist))
pntH=[PxH.real,PyH.real,PzH.real]
pntS=[PxS.real,PyS.real,PzS.real]
b.frame_alpha=.1
b.sphere_alpha=.2
b.markersize=.1
b.view =[-45,10]
b.add_points(pntH,'m')
b.add_points(pntS,'m')
b.point_color = list(colors)
pnt2=[Px2.real,Py2.real,Pz2.real]
b.add_points(pnt2,'m')
b.point_marker =['o']
b.point_size = [10]
b.show()
ax1= b.fig.add_axes([0.02 ,0.09 ,0.95 ,0.015])
cmap=mpl.cm.jet
cb1=mpl.colorbar.ColorbarBase(ax1,cmap = cmap,norm=nrm,orientation='horizontal')
cb1.set_label(r'$t\ [\mu s]$',fontsize = 10)
b.fig
b.fig.savefig('Grafica2bloch.pdf')


plt.figure (3)
plt.plot(tlistHS,dsdtSEAH, label = "SEAQT_A", color = 'red',linestyle='--')
plt.plot(tlistHS,dsdtQMH, label = "von Neumann_A", color = 'blue',linestyle='--')
plt.plot(tlistHS,dsdtSEAS, label = "SEAQT_B", color = 'red')
plt.plot(tlistHS,dsdtQMS, label = "von Neumann_B", color = 'blue')
plt.legend(loc="upper right")
plt.grid(True)
plt.grid(color = '0.5', linestyle = '--', linewidth = 1)
plt.xlabel(r"$time$", fontsize = 12, color = 'black')
plt.ylabel(r"$\frac{ds}{dt}$", fontsize = 15, color = 'black')
plt.title('Generacion de Entropia',fontsize = 12, color = 'black', verticalalignment = 'baseline', horizontalalignment = 'center')
plt.savefig('Grafica_entropy_generation.pdf')
plt.show()

plt.figure (4)
plt.plot(tlistHS,sSEAH, label = "SEAQT", color = 'red',linestyle='--')
plt.plot(tlistHS,sSEAS, label = "SEAQT", color = 'red')
plt.legend(loc="upper right")
plt.grid(True)
plt.grid(color = '0.5', linestyle = '--', linewidth = 1)
plt.xlabel(r"$time$", fontsize = 12, color = 'black')
plt.ylabel(r"$S$", fontsize = 15, color = 'black')
plt.title('Entropia Total',fontsize = 12, color = 'black', verticalalignment = 'baseline', horizontalalignment = 'center')
plt.savefig('Grafica_entropy.pdf')
plt.show()

plt.figure (5)
plt.plot(tlistHS,eigenva1H, label = r"$\lambda_1$", color = 'red')
plt.plot(tlistHS,eigenva2H, label = r"$\lambda_2$", color = 'blue')
plt.legend(loc="upper right")
plt.grid(True)
plt.grid(color = '0.5', linestyle = '--', linewidth = 1)
plt.xlabel(r"$time$", fontsize = 12, color = 'black')
plt.ylabel(r"$\lambda$", fontsize = 15, color = 'black')
plt.title('Eigenvalores',fontsize = 12, color = 'black', verticalalignment = 'baseline', horizontalalignment = 'center')
plt.savefig('Grafica_eigenvalues.pdf')
plt.show()

plt.figure (5)
plt.plot(tlistHS,EH, color = 'red')
plt.legend(loc="upper right")
plt.grid(True)
plt.xlim(33.56e-9,36.56e-9)
plt.grid(color = '0.5', linestyle = '--', linewidth = 1)
plt.xlabel(r"$time$", fontsize = 12, color = 'black')
plt.ylabel(r"$\lambda$", fontsize = 15, color = 'black')
plt.title('Energy',fontsize = 12, color = 'black', verticalalignment = 'baseline', horizontalalignment = 'center')
plt.savefig('Grafica_eigenvalues.pdf')
plt.show()

plt.figure (8)
plt.plot(tlistHS,PurityH, label = "A", color = 'red')
plt.plot(tlistHS,PurityS, label = "B", color = 'Blue')
plt.legend(loc="upper right")
#plt.xlim(-5e-9,185e-9)
#plt.ylim(.75,1)
plt.grid(True)
plt.grid(color = '0.5', linestyle = '--', linewidth = 1)
plt.xlabel(r"$t$", fontsize = 12, color = 'black')
plt.ylabel(r"$ Tr(\rho^2)$", fontsize = 15, color = 'black')
plt.title('Pureza de subsistemas',fontsize = 12, color = 'black', verticalalignment = 'baseline', horizontalalignment = 'center')
plt.savefig('Pureza.pdf')
plt.show()


plt.figure (2)
plt.figure 
plt.plot(tlistHS,PxH.real, label = "PxA", color = 'red',linestyle='--')
plt.plot(tlistHS,PyH.real, label = "PyA", color = 'blue',linestyle='--')
plt.plot(tlistHS,PzH.real, label = "PzA", color = 'green',linestyle='--')
plt.plot(tlist,Px, label = r"$P_x^A$", color = 'blue')
plt.plot(tlist,Py, label = r"$P_y^A$", color = 'red')
plt.plot(tlist,Pz, label = r"$P_z^A$", color = 'green')
#plt.ylim(-0.4,0.8)
plt.xlim(-5e-9,185e-9)
plt.legend(loc="upper right")
plt.grid(True)
plt.grid(color = '0.5', linestyle = '--', linewidth = 1)
plt.xlabel(r"$t$", fontsize = 12, color = 'black')
plt.ylabel(r"$Vector\ de\ Polarización\ P_A$", fontsize = 12, color = 'black')
plt.title('Evolución Qubit A',fontsize = 12, color = 'black', verticalalignment = 'baseline', horizontalalignment = 'center')
plt.savefig('PolarizacionA.pdf')
plt.show()

plt.figure (3)
plt.figure 
plt.plot(tlistHS,PxS.real, label = "PxB", color = 'red',linestyle='--')
plt.plot(tlistHS,PyS.real, label = "PyB", color = 'blue',linestyle='--')
plt.plot(tlistHS,PzS.real, label = "PzB", color = 'green',linestyle='--')
plt.plot(tlist,Px2, label = r"$P_x^B$", color = 'blue')
plt.plot(tlist,Py2, label = r"$P_y^B$", color = 'red')
plt.plot(tlist,Pz2, label = r"$P_z^B$", color = 'green')
plt.axhline(y=0.0, color='black')
plt.vlines(x=40.0e-9, ymin=-1, ymax=1)
#plt.ylim(-0.4,0.8)
plt.xlim(-5e-9,185e-9)
plt.legend(loc="upper right")
plt.grid(True)
plt.grid(color = '0.5', linestyle = '--', linewidth = 1)
plt.xlabel(r"$t$", fontsize = 12, color = 'black')
plt.ylabel(r"$Vector\ de\ Polarización\ P_B$", fontsize = 12, color = 'black')
plt.title('Evolución Qubit B',fontsize = 12, color = 'black', verticalalignment = 'baseline', horizontalalignment = 'center')
plt.savefig('PolarizacionB.pdf')
plt.show()


plt.figure (4)
plt.plot(tlist,Fac_Cor, color = 'blue')
plt.legend(loc="upper right")
#plt.ylim(-0.4,0.8)
plt.xlim(-5e-9,185e-9)
plt.grid(True)
plt.grid(color = '0.5', linestyle = '--', linewidth = 1)
plt.xlabel(r"$t$", fontsize = 12, color = 'black')
plt.ylabel(r"$||C||=Tr(CC^\dagger)$", fontsize = 12, color = 'black')
plt.title('Factor de Coherencia',fontsize = 12, color = 'black', verticalalignment = 'baseline', horizontalalignment = 'center')
plt.savefig('Factor_Coherencia.pdf')
plt.show()

plt.figure (5)
plt.plot(tlist,dsdtSEA, color = 'blue')
plt.legend(loc="upper right")
plt.xlim(-5e-9,185e-9)
#plt.ylim(.9,1.4)
plt.grid(True)
plt.grid(color = '0.5', linestyle = '--', linewidth = 1)
plt.xlabel(r"$t$", fontsize = 12, color = 'black')
plt.ylabel(r"$\frac{ds}{dt}$", fontsize = 15, color = 'black')
plt.title('Generación de Entropía',fontsize = 12, color = 'black', verticalalignment = 'baseline', horizontalalignment = 'center')
plt.savefig('Generacion_Entropia.pdf')
plt.show()

plt.figure (6)
plt.plot(tlist,sSEA, color = 'blue')
plt.legend(loc="upper right")
plt.xlim(-5e-9,185e-9)
#plt.ylim(.9,1.4)
plt.grid(True)
plt.grid(color = '0.5', linestyle = '--', linewidth = 1)
plt.xlabel(r"$t$", fontsize = 12, color = 'black')
plt.ylabel(r"$ \left. \langle S(\rho) \right. \rangle / k_B$", fontsize = 15, color = 'black')
plt.title('Entropía total',fontsize = 12, color = 'black', verticalalignment = 'baseline', horizontalalignment = 'center')
plt.savefig('Grafica_entropy_total.pdf')
plt.show()

plt.figure (7)
plt.plot(tlist,sSEA_A, label = "A", color = 'red')
plt.plot(tlist,sSEA_B, label = "B", color = 'Blue')
plt.legend(loc="upper right")
plt.xlim(-5e-9,185e-9)
#plt.ylim(0,1.4)
plt.grid(True)
plt.grid(color = '0.5', linestyle = '--', linewidth = 1)
plt.xlabel(r"$t$", fontsize = 12, color = 'black')
plt.ylabel(r"$ \left. \langle S(\rho) \right. \rangle / k_B$", fontsize = 15, color = 'black')
plt.title('Entropía total de subsistemas',fontsize = 12, color = 'black', verticalalignment = 'baseline', horizontalalignment = 'center')
plt.savefig('Grafica_entropy_subsistemas.pdf')
plt.show()


plt.figure (8)
plt.plot(tlist,Purity_A, label = "A", color = 'red')
plt.plot(tlist,Purity_B, label = "B", color = 'Blue')
plt.legend(loc="upper right")
plt.xlim(-5e-9,185e-9)
#plt.ylim(.75,1)
plt.grid(True)
plt.grid(color = '0.5', linestyle = '--', linewidth = 1)
plt.xlabel(r"$t$", fontsize = 12, color = 'black')
plt.ylabel(r"$ Tr(\rho^2)$", fontsize = 15, color = 'black')
plt.title('Pureza de subsistemas',fontsize = 12, color = 'black', verticalalignment = 'baseline', horizontalalignment = 'center')
plt.savefig('Pureza.pdf')
plt.show()


plt.figure (9)
plt.plot(tlist,sigmaAB, color = 'blue')
plt.legend(loc="upper right")
plt.xlim(-5e-9,185e-9)
#plt.ylim(0,.08)
plt.grid(True)
plt.grid(color = '0.5', linestyle = '--', linewidth = 1)
plt.xlabel(r"$t$", fontsize = 12, color = 'black')
plt.ylabel(r"$(\sigma_{AB}) $", fontsize = 15, color = 'black')
plt.title('Funcional de Correlación$\ \sigma_{AB}$',fontsize = 12, color = 'black', verticalalignment = 'baseline', horizontalalignment = 'center')
plt.savefig('Funcional_Correlacion.pdf')
plt.show()

plt.figure (10)
plt.plot(tlist,Concurrencia, color = 'blue')
plt.legend(loc="upper right")
plt.xlim(-5e-9,185e-9)
#plt.ylim(0,.08)
plt.grid(True)
plt.grid(color = '0.5', linestyle = '--', linewidth = 1)
plt.xlabel(r"$t$", fontsize = 12, color = 'black')
plt.ylabel(r"$C(\rho)$", fontsize = 15, color = 'black')
plt.title('Concurrencia C$',fontsize = 12, color = 'black', verticalalignment = 'baseline', horizontalalignment = 'center')
plt.savefig('Concurrencia.pdf')
plt.show()