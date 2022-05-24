function dydt = Funcion_Fermi_2Q(t,C)
w=.2675e+9;

tao=5
t_0=10
theta_G=((sqrt(pi)/2*tao)*exp(-((t-t_0)/tao)^2))
dydt=(1/i)*(C(1)*theta_G)


end 