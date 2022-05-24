function [drhodt]=SEAQ(t,rho)
format long g;
sigma3=[1 0;0 -1];
sigma1=[0 1;1 0];
rho=reshape(rho,2,2);
w=.2675;
hb=1;
kb=1;
tau=10;

tao=.1;
t0=34.5625;
t1=t0-tao/2;
t2=t0+tao/2;
thetaB=exp(-((t-t0)/tao)^2);
thetaG=(sqrt(pi)/(2*tao))*exp(-((t-t0)/tao)^2);

H=(1/2)*w*(1-thetaB)*sigma3+hb*thetaG*sigma1;
%H=(1/2)*hb*w*sigma3
gamma=trace(rho*H^2)-(trace(rho*H))^2;
alpha=(trace(rho*H)*trace(rho*H*logm(rho))-trace(rho*H^2)*trace(rho*logm(rho)))/gamma;
beta=(trace(rho*logm(rho))*trace(rho*H)-trace(rho*H*logm(rho)))/gamma;
D=(rho*logm(rho))+(alpha*rho)+((1/2)*beta*(H*rho+rho*H));
D=0
drho=(-(i/hb)*(H*rho-rho*H)-D/tau);
drhodt=reshape(drho,4,1);

end 