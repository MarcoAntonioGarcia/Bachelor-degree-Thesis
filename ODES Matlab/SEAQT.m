close all;
clc;
clear all;ana.docx
format long g

%Asignaci?n de Variables
sigma1=[0 1;1 0];

sigma2=[0 -1i;1i 0];
sigma3=[1 0;0 -1];
id=eye(2);
tspan=[0:.005:80];
r1=0.5;r2=0;r3=0.8;
w=.2675;hb=1;kb=1;tau=10;
%H=(1/2)*w*sigma3;
rho0=(1/2)*(r1*sigma1+r2*sigma2+r3*sigma3+id);
%rho0=reshape(rho0,4,1);
[t,rho]=ode45(@SEAQ,tspan,reshape(rho0,1,4));
    

rx(:,1)=rho(:,3)+rho(:,2);
ry(:,1)=(-1/1i)*(rho(:,2)-rho(:,3));
rz(:,1)=rho(:,1)-rho(:,4);
%p(:,n)=eig(reshape(rho(n,:),2,2));

%Componentes rx,ry,rz
for n=1:length(t)
    %s(n,1)=-kb*trace(reshape(rho(n,:),2,2)*logm(reshape(rho(n,:),2,2)));
    %E(n,1)=trace(H*reshape(rho(n,:),2,2));
    %rx(n,1)=rho(n,3)+rho(n,2);
    %ry(n,1)=(1/1i)*(rho(n,2)-rho(n,3));
    %rz(n,1)=rho(n,1)-rho(n,4);
    %p(:,n)=eig(reshape(rho(n,:),2,2));
    %%% ds/dt
    %gamma=trace(reshape(rho(n,:),2,2)*H^2)-(trace(reshape(rho(n,:),2,2)*H))^2;
    %alpha=(trace(reshape(rho(n,:),2,2)*H)*trace(reshape(rho(n,:),2,2)*H*logm(reshape(rho(n,:),2,2)))-trace(reshape(rho(n,:),2,2)*H^2)*trace(reshape(rho(n,:),2,2)*logm(reshape(rho(n,:),2,2))))/gamma;
    %beta=(trace(reshape(rho(n,:),2,2)*logm(reshape(rho(n,:),2,2)))*trace(reshape(rho(n,:),2,2)*H)-trace(reshape(rho(n,:),2,2)*H*logm(reshape(rho(n,:),2,2))))/gamma;
    %D=(reshape(rho(n,:),2,2)*logm(reshape(rho(n,:),2,2)))+(alpha*reshape(rho(n,:),2,2))+((1/2)*beta*(Anticommutator_JR(H,reshape(rho(n,:),2,2))));
    %Dg=((reshape(rho(n,:),2,2))^(1/2)*logm(reshape(rho(n,:),2,2)))+(alpha*(reshape(rho(n,:),2,2))^(1/2))+((reshape(rho(n,:),2,2))^(1/2)*H*beta);
    %dsdt(n,1)=(kb/tau)*trace(D'*D);
    %dsdtg(n,1)=(kb/tau)*trace(Dg'*Dg);
end
 
% %%%posici?n vs tiempo
figure
plot(t,rx,t,ry,t,rz)
grid on
title('Position')
xlabel('t/\tau')
ylabel('Position')
legend('r_x','r_y','r_z')

% %%%Trayecoria espiral
figure
hold on
grid on 
plot3(rx,ry,rz,'r')
axis([-1,1,-1,1,-1,1])
xlabel('r_x')
ylabel('r_y')
zlabel('r_z')
title('Bloch sphere')
[x,y,z]=sphere;
w=surf(x,y,z);
set(w, 'FaceAlpha', 0.3)
colormap summer
shading interp;
hold off

 
figure
hold on
grid on 
plot(s,E)
plot(sy,Ey)
title('Energy-Entropy')
xlabel('Entropy')
ylabel('Energy')
hold off

% %%%Entrop?a vs tiempo
figure
plot(t,s)
title('Entropy')
xlabel('t/\tau')
ylabel('sp')

% %%%Probabilidad de ocupaci?n
figure
plot(t,p)
title('Probabilidad de ocupaci?n')
xlabel('t/\tau')
ylabel('p_i')

% %%%Generaci?n de entrop?a
figure
hold on
plot(t,dsdt)
%plot(t,dsdtg)
title('Entropy Generation')
xlabel('t/\tau')
ylabel('ds/dt')