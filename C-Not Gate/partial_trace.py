def partial_traceA(rho,np):
    rhoA=np.zeros(((2,2)),complex)
    rhoA[0,0]=rho[0,0]+rho[1,1]
    rhoA[0,1]=rho[0,2]+rho[1,3]
    rhoA[1,0]=rho[2,0]+rho[3,1]
    rhoA[1,1]=rho[2,2]+rho[3,3]
    return(rhoA)
def partial_traceB(rho,np):    
    rhoB=np.zeros(((2,2)),complex)
    rhoB[0,0]=rho[0,0]+rho[2,2]
    rhoB[0,1]=rho[0,1]+rho[2,3]
    rhoB[1,0]=rho[1,0]+rho[3,2]
    rhoB[1,1]=rho[1,1]+rho[3,3]
    return(rhoB) 
    
    