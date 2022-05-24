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
from estados_iniciales import Hadamard_inicial
from estados_iniciales import Qubit_objetivo


#matrices de Pauli
sx=np.matrix(([[0.0, 1.0],[1.0, 0.0]]),complex)
sy=np.matrix(([[0.0, -1.0j],[1.0j, 0.0]]),complex)
sz=np.matrix(([[1.0, 0.0],[0.0, -1.0]]),complex)
ide=np.eye(2)

#bases para espacio de Hilbert dimencion 4
sx2=np.kron(sx,sx)     
sy2=np.kron(sy,sy)
sz2=np.kron(sz,sz)

#Caso='Resultados anteriores'
Caso='Resultados Circuito H-CNOT'

if Caso=='Resultados anteriores':
   Resultados_Montilla='H-CNOT'
#parametros de la compuerta
   tao=5;               #tiempo de accion de la compueta
   t0=10;               #pulso medio de la compuerta
   t1=t0-tao/2;            #tiempo de inicio de activacion de compuerta
   t2=t0+tao/2;            #tiempo de termino de activacion de compuerta
   w=.2675;             #frecuencia angular
   

#constantes fisicas utilizadas
   hb=1;                #constante modificada de Planck
   kb=1;                   #constante de Boltzman
   FA=0.0002*hb*w          #fuerza de acoplamiento entre subsistemas termino que quiere mantener acoplados los subsistemas...
   V=-FA*(sx2+sy2+sz2)     
   if Resultados_Montilla=='CNOT':
      tauA=40;             #tiempo disipativo A
      tauB=40;             #tiempo disipativo B

      #RESULTADOS DE MONTILLA
      #caso Controlled Not sin compuerta Hadamard iniciando con los estados |1> en el objetivo y |1> en el de control
      rhoAi=np.matrix(([[0.01, .0],[.0, .98]]),complex) #qubit #de control
      rhoBi=np.matrix(([[.02, 0],[0, .98]]),complex)  #qubit objetivo
    
   if Resultados_Montilla=='H-CNOT':
      tauA=10;             #tiempo disipativo A
      tauB=10;             #tiempo disipativo B
      #caso controlled not con compuerta Hadamard en el qubit objetivo   
      rhoAi=np.matrix(([[0.5, .5],[.5, .5]]),complex) #qubit #de control
      rhoBi=np.matrix(([[.98, 0],[0, .02]]),complex)  #qubit objetivo

       
if Caso=='Resultados Circuito H-CNOT':
    tauA=40;             #tiempo disipativo A
    tauB=40;             #tiempo disipativo B

#parametros de la compuerta
    tao=5;              #tiempo de accion de la compueta
    t0=10;               #pulso medio de la compuerta
    t1=t0-tao/2;            #tiempo de inicio de activacion de compuerta
    t2=t0+tao/2;            #tiempo de termino de activacion de compuerta
    #w=1
    w=.2675;              #frecuencia angular

#constantes fisicas utilizadas
    hb=1;                #constante modificada de Planck
    kb=1;                   #constante de Boltzman
    FA=0.0002*hb*w          #fuerza de acoplamiento entre subsistemas termino que quiere mantener acoplados los subsistemas...
    V=-FA*(sx2+sy2+sz2)  
 
   #polarización para qubit A (Control) Esto define el estado inicial de la compuerta Hadamard bajo la dinamica de SEAQT
   #la H es de 'Hadamard' en todo el codigo y la función generadora de estados iniciales
    r1H=0.001;
    r2H=0;
    r3H=.999;

#polarización para qubit B (Objetivo) Esto define el estado inicial precesando bajo el marco de SEAQT
#la S es de 'sin compuerta'
    r1S=0.001;
    r2S=0;
    r3S=.999;

#parametros para los estados iniciales
    tauH=40         #tiempo disipativo de la compuerta Hadamard
    tauS=40        #tiempo disipativo del qubit objetivo 
    tHS = 40        #tiempo de evolución de los estados iniciales
    dtHS = .1       #intervalos de integración numérica

#RESULTADOS DEL CIRCUITO H-CNOT CON TODAS LAS COMPUERTAS DEL PROTOCOLO BAJO EL MARCO DE SEAQT
    rhoAi=Hadamard_inicial(r1H,r2H,r3H,tauH,tHS,dtHS)
    rhoBi=Qubit_objetivo(r1S,r2S,r3S,tauS,tHS,dtHS)


################################################Compuerta CNOT#######################################################
#le evolución de la compuerta CNOT tiene como qubit de control una compuerta Hadamard que ya ha sido evolucionada
#con el marco de SEAQT y como qubit objetivo un estado precesando que ya fue evolucionado en el marco de SEAQT

rhoi=np.kron(rhoAi,rhoBi)
rho1=np.asarray(rhoi).reshape(1,16)

   

#Hamiltonianos de subsistemas
HA=-(1/2)*(hb*w)*sz
HB=-(1/2)*(hb*w)*sz

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
    
    lnrho=sp.linalg.logm(rho)
    lnrhoA=partial_traceA((np.dot(np.kron(ide,rhoB),lnrho)),np)
    lnrhoB=partial_traceB((np.dot(np.kron(rhoA,ide),lnrho)),np)
    
    thetaB=math.exp(-((t-t0)/tao)**2);
    thetaG=(math.sqrt(math.pi)/(2*tao))*math.exp(-((t-t0)/tao)**2);
    H=(1-thetaB)*H1+hb*thetaG*C_NOT
    
    HA=partial_traceA((np.dot(np.kron(ide,rhoB) ,H)),np)
    HB=partial_traceB((np.dot(np.kron(rhoA,ide) ,H)),np)
    
    gammaA=np.trace(np.dot(rhoA,np.dot(HA,HA)))-(np.trace(np.dot(rhoA,HA)))**2;
    alphaA=(np.trace(np.dot(rhoA,HA))*(1/2)*np.trace(np.dot(rhoA,(np.dot(HA,lnrhoA)+np.dot(lnrhoA,HA))))-np.trace(np.dot(rhoA,np.dot(HA,HA)))*np.trace(np.dot(rhoA,lnrhoA)))/gammaA;
    betaA=(np.trace(np.dot(rhoA,lnrhoA))*np.trace(np.dot(rhoA,HA))-(1/2)*np.trace(np.dot(rhoA,(np.dot(HA,lnrhoA)+np.dot(lnrhoA,HA)))))/gammaA;
    DA=(1/2)*((np.dot(rhoA,lnrhoA))+(np.dot(lnrhoA,rhoA)))+(alphaA*rhoA)+((1/2)*betaA*(np.dot(rhoA,HA)+np.dot(HA,rhoA)));

    gammaB=np.trace(np.dot(rhoB,np.dot(HB,HB)))-(np.trace(np.dot(rhoB,HB)))**2;
    alphaB=(np.trace(np.dot(rhoB,HB))*(1/2)*np.trace(np.dot(rhoB,(np.dot(HB,lnrhoB)+np.dot(lnrhoB,HB))))-np.trace(np.dot(rhoB,np.dot(HB,HB)))*np.trace(np.dot(rhoB,lnrhoB)))/gammaB;
    betaB=(np.trace(np.dot(rhoB,lnrhoB))*np.trace(np.dot(rhoB,HB))-(1/2)*np.trace(np.dot(rhoB,(np.dot(HB,lnrhoB)+np.dot(lnrhoB,HB)))))/gammaB;
    DB=(1/2)*((np.dot(rhoB,lnrhoB))+(np.dot(lnrhoB,rhoB)))+(alphaB*rhoB)+((1/2)*betaB*(np.dot(rhoB,HB)+np.dot(HB,rhoB))); 
    D=((1/tauA)*(np.kron(DA,rhoB)))+((1/tauB)*(np.kron(rhoA,DB)))
    #D=0
    drho=-(1j/hb)*(np.dot(H,rho)-np.dot(rho,H))-D;
    
    return drho.reshape(1,16)

r = complex_ode(f)
r.set_initial_value(rho1, 0)

t = 180
dt = .1
i=00

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
zeros_FD=np.zeros(((int(t/dt))),float)
Concurrencia=np.zeros(((int(t/dt))),float)
drho1=np.zeros(((int(t/dt),16)),float)
rho=np.zeros(((int(t/dt),4,4)),float)
eigenva1H=np.zeros(int(t/dt),float)
eigenva2H=np.zeros(int(t/dt),float)

Energy_B=np.zeros(int(t/dt),float)


while r.successful() and (i+dt)*dt <= t:
    r.integrate(r.t+dt)
    rho=r.y.reshape(4,4)

    rhoA=partial_traceA(rho,np)
    rhoB=partial_traceB(rho,np)
    
    thetaB=math.exp(-((t-t0)/tao)**2);
    thetaG=(math.sqrt(math.pi)/(2*tao))*math.exp(-((t-t0)/tao)**2);
    H=(1-thetaB)*H1+hb*thetaG*C_NOT     #si quitamos V tambien funciona
    
    HA=partial_traceA((np.dot(np.kron(ide,rhoB) ,H)),np)
    HB=partial_traceB((np.dot(np.kron(rhoA,ide) ,H)),np)
    
    #HA=(1/2)*(hb*w)*sz
    #HB=(1/2)*(hb*w)*sz
    
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
    Energy_B[i]=np.trace(np.dot(HB,rhoB))
    
    eig= np.linalg.eig(rho)
    eig2=eig[0]
    eigenva1H[i]=eig2[0]
    eigenva2H[i]=eig2[1]
    #EH[i]=np.trace(np.dot(H,rho))

    
    #concurrencia
    rho_concurrencia=np.dot(np.dot(sy2,np.conjugate(rho)),sy2)
    rho_raiz=sp.linalg.sqrtm(rho)
    R=sp.linalg.sqrtm(np.dot(np.dot(rho_raiz,rho_concurrencia),rho_raiz))
    R_lamda=sorted(np.linalg.eigh(R)[0])
    Concurrencia[i]=R_lamda[3]-R_lamda[2]-R_lamda[1]-R_lamda[0]
    zeros_FD[i]=rho[0,1]*np.conjugate(rho[0,1])+rho[0,2]*np.conjugate(rho[0,2])+rho[0,3]*np.conjugate(rho[0,3])+rho[1,0]*np.conjugate(rho[1,0])+rho[1,2]*np.conjugate(rho[1,2])+rho[1,3]*np.conjugate(rho[1,3])+rho[2,0]*np.conjugate(rho[2,0])+rho[2,1]*np.conjugate(rho[2,1])+rho[2,3]*np.conjugate(rho[2,3])+rho[3,0]*np.conjugate(rho[3,0])+rho[3,1]*np.conjugate(rho[3,1])+rho[3,2]*np.conjugate(rho[3,2])
    
    dsdtSEA[i]=-(np.trace(np.dot(drho,sp.linalg.logm(rho)))-np.trace(drho))/w
    
    time[i]=dt*i
    Px[i]=rhoA[0,1]+rhoA[1,0]
    Py[i]=(-1/(1j))*(rhoA[0,1]-rhoA[1,0])
    Pz[i]=(rhoA[0,0]-rhoA[1,1])
    
    Px2[i]=rhoB[0,1]+rhoB[1,0]
    Py2[i]=(-1/(1j))*(rhoB[0,1]-rhoB[1,0])
    Pz2[i]=(rhoB[0,0]-rhoB[1,1]) 
    
    i=i+1
tlist = np.linspace(0,t,t/dt)      


################################Esfera de Bloch################################
plt.figure (1)
b = qt.Bloch()  
nrm = mpl.colors.Normalize(0,t)
colors = cm.jet(nrm(tlist))
pnt=[Px.real,Py.real,Pz.real]
b.frame_alpha=.1
b.sphere_alpha=.2
b.markersize=.1
b.view =[-45,10]
b.view =[110,25]
#b.add_points(pnt,'m')
b.point_color = list(colors)
b.point_marker =['o']
b.point_size = [10]
vecAi =[Px.real[0][0],Py.real[0][0],Pz.real[0][0]]
b.add_vectors(vecAi)
vecAf =[Px.real[-1][0],Py.real[-1][0],Pz.real[-1][0]]
b.add_vectors(vecAf)
pnt2=[Px2.real,Py2.real,Pz2.real]
b.add_points(pnt2,'m')
vecBi =[Px2.real[0][0],Py2.real[0][0],Pz2.real[0][0]]
b.add_vectors(vecBi)
vecBf =[Px2.real[-1][0],Py2.real[-1][0],Pz2.real[-1][0]]
b.add_vectors(vecBf)
b.show()
ax1= b.fig.add_axes([0.02 ,0.09 ,0.95 ,0.015])
cmap=mpl.cm.jet
cb1=mpl.colorbar.ColorbarBase(ax1,cmap = cmap,norm=nrm,orientation='horizontal')
cb1.set_label(r'$t\ [s]$',fontsize = 10)
b.fig
b.fig.savefig('Grafica2bloch2.pdf')


##########################Vector de Polarizacion A#############################
plt.figure (2)
plt.figure 
plt.plot(tlist,Px, label = r"$P_x^A$", color = 'blue')
plt.plot(tlist,Py, label = r"$P_y^A$", color = 'red')
plt.plot(tlist,Pz, label = r"$P_z^A$", color = 'green')
#plt.xlim(-5e-9,185e-9)
plt.legend(loc="upper right")
#plt.grid(True)
#plt.grid(color = '0.5', linestyle = '--', linewidth = 1)
plt.xlabel(r"$U_t$", fontsize = 12, color = 'black')
plt.ylabel(r"$Vector\ de\ Polarización\ P_A$", fontsize = 12, color = 'black')
#plt.title('Evolución Qubit A',fontsize = 12, color = 'black', verticalalignment = 'baseline', horizontalalignment = 'center')
plt.savefig('PolarizacionA.pdf')
plt.show()


##########################Vector de Polarizacion B#############################
plt.figure (3)
plt.figure 
plt.plot(tlist,Px2, label = r"$P_x^B$", color = 'blue')
plt.plot(tlist,Py2, label = r"$P_y^B$", color = 'red')
plt.plot(tlist,Pz2, label = r"$P_z^B$", color = 'green')
#plt.xlim(-5e-9,185e-9)
plt.legend(loc="upper right")
#plt.grid(True)
#plt.grid(color = '0.5', linestyle = '--', linewidth = 1)
plt.xlabel(r"$U_t$", fontsize = 12, color = 'black')
plt.ylabel(r"$Vector\ de\ Polarización\ P_B$", fontsize = 12, color = 'black')
#plt.title('Evolución Qubit B',fontsize = 12, color = 'black', verticalalignment = 'baseline', horizontalalignment = 'center')
plt.savefig('PolarizacionB.pdf')
plt.show()


###############################Factor de Coherencia############################
plt.figure (4)
plt.plot(tlist,Fac_Cor, color = 'blue')
plt.legend(loc="upper right")
#plt.xlim(-5e-9,185e-9)
#plt.grid(True)
#plt.grid(color = '0.5', linestyle = '--', linewidth = 1)
plt.xlabel(r"$U_t$", fontsize = 12, color = 'black')
plt.ylabel(r"$||C||=Tr(CC^\dagger)$", fontsize = 12, color = 'black')
#plt.title('Factor de Coherencia',fontsize = 12, color = 'black', verticalalignment = 'baseline', horizontalalignment = 'center')
plt.savefig('Factor_Coherencia.pdf')
plt.show()


############################degeracion de Entropia#############################
plt.figure (5)
plt.plot(tlist,dsdtSEA, color = 'blue')
plt.legend(loc="upper right")
#plt.xlim(-5e-9,185e-9)
plt.ylim(.9,1.4)
plt.grid(True)
plt.grid(color = '0.5', linestyle = '--', linewidth = 1)
plt.xlabel(r"$U_t$", fontsize = 12, color = 'black')
plt.ylabel(r"$\dot{S}(\rho) / k_B$", fontsize = 15, color = 'black')
plt.title('Generación de Entropía',fontsize = 12, color = 'black', verticalalignment = 'baseline', horizontalalignment = 'center')
plt.savefig('Generacion_Entropia.pdf')
plt.show()


###############################Entropia Percibida##############################
plt.figure (6)
plt.plot(tlist,sSEA, color = 'blue')
plt.legend(loc="upper right")
plt.xlim(-5e-9,185e-9)
plt.ylim(.9,1.4)
plt.grid(True)
plt.grid(color = '0.5', linestyle = '--', linewidth = 1)
plt.xlabel(r"$U_t$", fontsize = 12, color = 'black')
plt.ylabel(r"$S(\rho) / k_B$", fontsize = 15, color = 'black')
plt.title('Entropía total',fontsize = 12, color = 'black', verticalalignment = 'baseline', horizontalalignment = 'center')
plt.savefig('Grafica_entropy_total.pdf')
plt.show()


##########################Entropia percibida de subsistemas####################
plt.figure (7)
plt.plot(tlist,sSEA_A, label = "A", color = 'red')
plt.plot(tlist,sSEA_B, label = "B", color = 'Blue')
plt.legend(loc="upper right")
plt.xlim(-5e-9,185e-9)
plt.ylim(0,1.4)
plt.grid(True)
plt.grid(color = '0.5', linestyle = '--', linewidth = 1)
plt.xlabel(r"$U_t$", fontsize = 12, color = 'black')
plt.ylabel(r"$S(\rho) / k_B$", fontsize = 15, color = 'black')
plt.title('Entropía total de subsistemas',fontsize = 12, color = 'black', verticalalignment = 'baseline', horizontalalignment = 'center')
plt.savefig('Grafica_entropy_subsistemas.pdf')
plt.show()


#############################Pureza de subsistemas#############################
plt.figure (8)
plt.plot(tlist,Purity_A, label = "A", color = 'red')
plt.plot(tlist,Purity_B, label = "B", color = 'Blue')
plt.legend(loc="upper right")
plt.xlim(-5e-9,185e-9)
plt.ylim(.75,1)
plt.grid(True)
plt.grid(color = '0.5', linestyle = '--', linewidth = 1)
plt.xlabel(r"$U_t$", fontsize = 12, color = 'black')
plt.ylabel(r"$ Tr(\rho^2)$", fontsize = 15, color = 'black')
plt.title('Pureza de subsistemas',fontsize = 12, color = 'black', verticalalignment = 'baseline', horizontalalignment = 'center')
plt.savefig('Pureza.pdf')
plt.show()


###########################Funcional de Correlacion#############################
plt.figure (9)
plt.plot(tlist,sigmaAB, color = 'blue')
plt.legend(loc="upper right")
plt.xlim(-5e-9,185e-9)
plt.ylim(0,.08)
plt.grid(color = '0.5', linestyle = '--', linewidth = 1)
plt.grid(True)
plt.xlabel(r"$U_t$", fontsize = 12, color = 'black')
plt.ylabel(r"$\sigma_{AB} $", fontsize = 15, color = 'black')
plt.title('Funcional de Correlación$\ \sigma_{AB}$',fontsize = 12, color = 'black', verticalalignment = 'baseline', horizontalalignment = 'center')
plt.savefig('Funcional_Correlacion.pdf')
plt.show()



##################################Concurrencia#################################
plt.figure (10)
plt.plot(tlist,Concurrencia, color = 'blue')
plt.legend(loc="upper right")
plt.xlim(-5e-9,185e-9)
plt.ylim(0,.08)
plt.grid(color = '0.5', linestyle = '--', linewidth = 1)
plt.grid(True)
plt.xlabel(r"$U_t$", fontsize = 12, color = 'black')
plt.ylabel(r"$C(\rho)$", fontsize = 15, color = 'black')
plt.title('Concurrencia',fontsize = 12, color = 'black', verticalalignment = 'baseline', horizontalalignment = 'center')
plt.savefig('Concurrencia.pdf')
plt.show()


















