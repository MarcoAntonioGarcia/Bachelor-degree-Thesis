close all
clc
clear all
format long 

ti=10
tf=10.5
dt=.001

[t,C] = ode45(@Funcion_Fermi,[ti:dt:tf],[1;0]);

C1=(C(:,1))
C1C=conj(C1)
C2=(C(:,2))
C2C=conj(C2)

for i=1: length(C1)
    Norma(i)=sqrt(C1(i)*C1C(i)+C2(i)*C2C(i));
    alpha(i)=C1(i)/Norma(i);
    beta(i)=C2(i)/Norma(i);
    Probability_alpha(i)=alpha(i)*conj(alpha(i));
    Probability_beta(i)=beta(i)*conj(beta(i));
    Probability_total(i)=Probability_alpha(i)+Probability_beta(i);
end 

time=ti:dt:tf-dt;
Gamma=(diff(Probability_beta)/(dt))
Q_alpha = trapz(t,Probability_alpha)
Q_beta = trapz(t,Probability_beta)
Gamma_integrado=trapz(time,Gamma)
GammaP=(1/(tf-ti))*Gamma_integrado
t_D=GammaP^-1

figure (4)
plot(t,Probability_alpha, t,Probability_beta,time,Gamma,'-')
title('Solution of C_1 and C_2');
xlabel('t');
ylabel('Solution C');
legend('P_{alpha}','P_{beta}','Gamma_{alpha}')
grid on 








