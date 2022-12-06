import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import sympy as sb

##MMS

t,r=sb.symbols("t,r")
T_0,t0=sb.symbols("T0,t0")
h,k,e=sb.symbols("h,k,e")
Eq_MMS=T_0*sb.exp(t/t0)*sb.sin(sb.pi*r/e)

T0,t0=15,7200
h,k,e=4,10,6e-3

MMS=Eq_MMS.diff(t)-k/h*(Eq_MMS.diff(r,2)+(1/r)*Eq_MMS.diff(r))
def f_MMS(r,t,T0,t0,e):
    return T0*np.exp(t/t0)*np.sin(np.pi*r/e)
#f_source=sb.lambdify([r,t,T_0,t0,h,k,e],MMS,"numpy")

Nx=10

Vr = np.linspace(30e-3, 36e-3, Nx+1)[1:-1]

# Instants discrétisés
dt = 1.0
Vt = np.arange(0,7200+0.01*dt,dt)

SoluMMS = np.zeros((len(Vr),len(Vt)))
print(SoluMMS)
for i_t,tMMS in enumerate(Vt) :
    print(f_MMS(Vr,tMMS,T0,t0,e))
    SoluMMS[:,i_t] = f_MMS(Vr,tMMS,T0,t0,e)