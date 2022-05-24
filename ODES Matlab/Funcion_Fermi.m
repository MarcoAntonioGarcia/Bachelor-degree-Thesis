function dydt = Funcion_Fermi(t,C)
w=.2675;
t_0=10.25
tao=.5
theta_G=(sqrt(pi)/2*tao)*exp(-((t-t_0)/tao)^2)
  
%dydt = [(1/i)*(C(2)*exp(i*w*t))*theta_G; (1/i)*(C(1)*exp(-i*w*t))*theta_G]
%Hadamard:
dydt = [(1/(i*sqrt(2)))*theta_G*(C(2)*exp(-i*w*t)+C(1)); (1/(i*sqrt(2)))*theta_G*(C(1)*exp(i*w*t)-C(2))]
end 

