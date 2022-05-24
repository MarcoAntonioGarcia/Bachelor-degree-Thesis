close all
clc
clear all
format long 

ti=10
tf=15
dt=0.1

[t,C] = ode45(@Funcion_Fermi_2Q,[ti:dt:tf],.9);

C1=(C(:,1))
C1C=conj(C1)

for i=1: length(C1)
    Norma(i)=sqrt(C1(i)*C1C(i));
    alpha(i)=C1(i)/Norma(i);
    Probability_alpha(i)=alpha(i)*conj(alpha(i));
    %Probability_total(i)=Probability_alpha(i)+Probability_beta(i);
end 

time=ti:dt:tf-dt;
Gamma=diff(Probability_alpha)/(dt)
Q_alpha = trapz(t,Probability_alpha)
%Q_beta = trapz(t,Probability_beta)
Gamma_integrado=trapz(time,Gamma)
GammaP=(1/(tf-ti))*Gamma_integrado
t_D=GammaP^-1




