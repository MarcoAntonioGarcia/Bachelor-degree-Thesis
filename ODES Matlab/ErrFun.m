close all;
clc;
clear all;
warning off;
format long g

t= 0:0.1:30
tao=.5
t0=10
t1=t0-tao/2
t2=t0+tao/2



for i=1:length(t)
    
    y1 =(1/2)*(erf(((t(i)-t1)/tao))-erf(((t(i)-t2)/tao)));
    y2 =(1/2)*(erf(((t0-t1)/tao))-erf(((t0-t2)/tao)));
    y(i)=y1/y2;
    
    r(i) =exp(-((t(i)-t0)/tao)^2);
    
end
figure (1)
plot(t,r)
grid on
title('CDF')
xlabel('t')
ylabel('error funcion')

figure (2)
plot(t,y)
grid on
title('CDF')
xlabel('t')
ylabel('error funcion')