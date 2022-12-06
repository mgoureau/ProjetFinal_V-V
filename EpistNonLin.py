import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
#from PropaIncertitude import LHSppf #Valeur de l'épaisseur de MCP
from sklearn import linear_model

T0 = 7.0 # Température initiale de la housse.
Text = 35.0 # Température extérieure

eta_vin = 400.0
#eta_air = 4.33

cp_vin = 3963
h=0.174 #hauteur bouteille en m
d=0.076 #diamètre bouteill en m
r_eau = d/2
V=np.pi*r_eau**2*h # (d/2)
S_B_i = 2*np.pi*r_eau*h
S_g = S_B_i
rho_v = 1000
Teau = 15 # Température initiale du eau

service = 2*3600.0 #Durée du calcul : 2eures

# la résistance thermique équivalente est en [m]/[W/K/m] = K/[W/m²]
# (température divisée par densité surfacique de flux)
# np.log(r2/r1)*(r1+r2)/2 / lambd ~ (r2-r1)/lambd

#e_MCP = LHSppf[0]
e_MCP=6 #mm


eta_arr = np.linspace(4,5,100)
T_arr = np.zeros(len(eta_arr))
for i_eta,eta_air in enumerate(eta_arr):
    print(eta_air)

    def resist_thq_cyl(r1,r2,k,couche,verbose=False):
        Req = (r2-r1)/k
        if verbose :
            print(couche)
            print(f"Résistance thermique ~ {1e6*Req:.2f} K.mm²/W")
        return Req

    # Récupérer les valeurs du cas linéaire 
    eta_g = eta_vin #Convection au niveau de l'eau
    eta_d = eta_air #Convection au niveau de l'air

    #Bouteille
    # Récupérer les valeurs du cas linéaire
    lamb_verre = 1.05    # 1.05
    rho_verre  = 2500 # 2500
    cp_verre   = 720   # 720
    r_verre    = 39.5e-3 # 39.5e-3
    #Néoprène
    lamb_neo = 0.192   # 0.192
    rho_neo  = 1230 # 1230
    cp_neo   = 2176  # 2176
    # def des r
    r_neo1   =   40e-3 # 40e-3
    r_plast1 = 40.5e-3 # 40.5e-3
    r_MCP    = r_plast1 + e_MCP*1e-3 # 46.5e-3
    r_plast2 =   r_plast1 + e_MCP*1e-3 + 0.5e-3 # 47e-3
    r_neo2   = r_plast1 + e_MCP*1e-3 + 1e-3 # 47.5e-3
    #Plastique
    lamb_plast = 0.26   # 0.26
    rho_plast  = 1200 # 1200
    cp_plast   = 1500  # 1500

    #MCP :
    from Hprime_comED import (h_pr_v, h_pr_r, k, T_l_v, T_r_v, T_l_r, T_r_r, \
                            T_min_r, T_max_r, T_min_v, T_max_v, rho_s_v, \
                            rho_l_v, rho_s_r, rho_l_r, cp_MCP)


    print(f"*** MCP Non linéaire ***\n")

    # Température extérieure constante et égale à Text
    T_air = Text # 25.0

    #Discrétisation
    Nx = 5
    e  = (r_MCP - r_plast1)
    dr = e/Nx

    #Résistance thermique
    r_th_bout = resist_thq_cyl(r_verre,r_neo1,lamb_verre,"verre", verbose=False)
    r_th_neo1   = resist_thq_cyl(r_neo1,r_plast1,lamb_verre,"néo 1", verbose=False)
    r_th_neo2   = resist_thq_cyl(r_plast2,r_neo2,lamb_verre,"néo 2", verbose=False)
    r_th_plast1 = resist_thq_cyl(r_neo1,r_plast1,lamb_verre,"plast 1", verbose=False)
    r_th_plast2 = resist_thq_cyl(r_MCP,r_plast2,lamb_verre,"plast 2", verbose=False)

    Reqg = r_th_bout + r_th_neo1 + r_th_plast1
    #print(f"Résistance thermique équivalente à gauche ~ {1e6*Reqg:.2f} K.mm²/W")
    Reqd = r_th_neo2 + r_th_plast2
    #print(f"Résistance thermique équivalente à droite ~ {1e6*Reqd:.2f} K.mm²/W")

    #Récupération de h'
    def alpha(T):
        #Modèle non linéaire
        return k/h_pr_r(T)

    #Mise sous forme de problème de Cauchy
    Vr = np.linspace(r_plast1, r_MCP, Nx+1)[1:-1]

    # Instants discrétisés
    dt = 1.0
    Vt = np.arange(0,service+0.01*dt,dt)

    # Coefficients de convection équivalents
    # Formulaire page 5
    eta_eqg=eta_g/(1+eta_g*Reqg)
    eta_eqd=eta_d/(1+eta_d*Reqd)

    # Coefficients de convection équivalents
    # Formulaire page 4
    beta = k/(2*dr)
    # Convection vers l'intérieur (à gauche)
    dg = beta/(3*beta+eta_eqg)
    gg = eta_eqg/(3*beta+eta_eqg)
    # Convection vers l'extérieur (à droite)
    dd = beta/(3*beta+eta_eqd)
    gd = eta_eqd/(3*beta+eta_eqd)

    Ar = -0.5/(dr*Vr) + 1.0/dr**2
    b = -2/dr**2
    Cr = 0.5/(dr*Vr) + 1.0/dr**2

    Tinit = T0 * np.ones(Nx) # Récupérer la température initiale linéaire
    Tinit[0] = Teau          # Idem

    def FNonLin(Y,t):
        Y = np.array(Y) # Sécurité
        Yprime = np.zeros_like(Y)
        # Première ligne i=0 # Cavité de l'eau
        Tgstar = 4*dg*Y[1] - dg*Y[2] + gg*Y[0] # Formulaire page 4
        Yprime[0] = 1/(rho_v*cp_vin*V)*S_g*eta_eqg*(Tgstar-Y[0])
        # Deuxième ligne i=1 # Premier nœud du MCP
        Yprime[1] = alpha(Y[1])*(Ar[0]*Tgstar+b*Y[1]+Cr[0]*Y[2])
        #Lignes de i=2 à i=Nx-2
        for n in range(2, Nx-1):
            Yprime[n] = alpha(Y[n])*(Ar[n-1]*Y[n-1]+b*Y[n]+Cr[n-1]*Y[n+1])
        #Dernière ligne i=Nx-1
        Tdstar = -dd*Y[-2] + 4*dd*Y[-1] + gd*T_air # Formulaire page 4
        Yprime[-1] = alpha(Y[-1])*(Ar[-1]*Y[-2]+b*Y[-1]+Cr[-1]*Tdstar)
        return Yprime

    # Mémorisation des résultats
    soluNonLin=odeint(FNonLin,Tinit,Vt)

    print("Temp eau après 2h : ",soluNonLin[-1,0])
    T_arr[i_eta] = soluNonLin[-1,0]
    

plt.figure("Résultats de la simulation",figsize=(16,7))
plt.plot(eta_arr,T_arr,".",label="T° eau")

plt.xlabel("h air")
plt.ylabel("Température eau (°C)")


plt.title("Température de l'eau en fonction du coefficient de convection de l'air")
plt.legend()
plt.show()

