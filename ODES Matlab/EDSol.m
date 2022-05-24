clear all
close all
clc
format long

syms a y(t)
eqn = diff(y,t) == a*y;
dsolve(eqn)

syms y(t) a
eqn = diff(y,t,2) == a*y;
ySol(t) = dsolve(eqn)

syms y(t) a
eqn = diff(y,t) == a*y;
cond = y(0) == 5;
ySol(t) = dsolve(eqn,cond)



syms x1(t) x2(t) h w 
eqns = [diff(x1,t) == (-i/h)*(exp(-i*w*t))*x2, diff(x2,t) == (-i/h)*(exp(i*w*t))*x1];
[x1Sol(t) x2Sol(t)]= dsolve(eqns, 'IgnoreAnalyticConstraints', false)

%[xSol(t) ySol(t)]=dsolve('Dx=-x+y','Dy=-2*x-4*y+exp(t)','t')