# Version 2 - 2022, Mai, 16
# Project : MCP
# Author : Lunaï PRAWERMAN & Aurélien COUPEAU
# Institution : ENSAM
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
########################################################################
## PROJET MCP : BOUTEILLE MAINTENUE EN TEMPERATURE AVEC HOUSSE MCP #####
########################################################################
# Pour aller chercher les modules de SysTherLin :
import os,sys
rel_sys_ther_lin_path = "../../../../SysTherLin" # Chemin relatif ÉD
abs_sys_ther_lin_path = os.path.abspath(rel_sys_ther_lin_path)
sys.path.append(abs_sys_ther_lin_path)
# ----------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from Couches_conductrices import (CoucheConductriceCylindrique,\
                                  CoucheConductriceSpherique,\
                                  CoucheConductrice,Multicouche)
from Systemes_thermiques_lineaires import (Cavite,\
                                           SystemeThermiqueLineaire)
from scipy.integrate import odeint

## 1 - Définition des couches conductrices
T0 = 7.0 # Température initiale de la housse.
# La housse étant stockée dans un frigo on considère que toutes ses couches
# sont à la température initiale T0
Text = 25.0 # Température extérieure
#Par la suite on impose une temperature notre bouteille de 11°C pour avoir
# une dégustation optimale  
Tverre = 11.0 # Créer un paramètre
## 1.2 - Couches cylindriques de la bouteille et de la housse
verre  = CoucheConductriceCylindrique( 1.05, 2500 ,  720,   38e-3,39.5e-3, Tverre)
revet1 = CoucheConductriceCylindrique(0.192, 1230 , 2176, 39.5e-3,  40e-3, T0)
plast1 = CoucheConductriceCylindrique( 0.26, 1200 , 1500,   40e-3,40.5e-3, T0)
mcp    = CoucheConductriceCylindrique(  0.2,  880 , 2000, 40.5e-3,  43.5e-3, T0)  #On choisit le MCP RT15 pour usage rosé
plast2 = CoucheConductriceCylindrique( 0.26, 1200 , 1500,   43.5e-3,44e-3, T0)
revet2 = CoucheConductriceCylindrique(0.192, 1230 , 2176, 44e-3,  44.5e-3, T0)
# Vérification de la cohérence des paramètres
print(f"Rayon intérieur de la bouteille : {verre.Rmin*100:.2f} cm")
print(f"Épaisseur du MCP : {(mcp.Rmax-mcp.Rmin)*100:.2f} cm")
# Faire afficher les constantes de temps :
msg = "**** Constantes de temps :"
for couche,label in zip( (verre,revet1,plast1,mcp,plast2,revet2), \
                         ("verre","revet1","plast1","mcp","plast2","revet2") ):
    msg += f"\n\t{label}\t~ {couche.tau:.2f} s"
print(msg)
## 2 - Définition des multicouches avec conditions limites
# 2.1 Cloche hémisphérique
coque = Multicouche([verre,revet1,plast1,mcp,plast2,revet2])
eta_vin = 400.0 # Pour pouvoir être utilisé plus bas
coque.definir_CL("G","Convection",eta_vin) # Côté intérieur
#coque.definir_CL("G","Dirichlet",lambda s : 11.0/s) # Côté intérieur
eta_air = 4.33  # Idem
coque.definir_CL("D","Convection",eta_air) # Côté extérieur
# Cas extrême : Condition de Dirichlet (correspond à un coef. de convection infini)
# On ne définit pas le signal à ce moment-là : ", lambda s : Text/s'" en trop.
#coque.definir_CL("D","Dirichlet") # Côté extérieur à la température Text
print(coque)

## 3 - Définition des cavités
cp_vin = 3963
h=0.174 #hauteur bouteille en m
d=0.076 #diamètre bouteill en m
r_vin = d/2
print(f"(verre.Rmin,r_vin) [cm] -> " + \
      f"({100*verre.Rmin:.2f},{100*r_vin:.2f})")

rho_ethanol = 789
V=np.pi*r_vin**2*h # (d/2) = r_vin => remplacé partout
S_B_i = 2*np.pi*r_vin*h
S_g = S_B_i
rho_v = rho_ethanol*0.14+1000*(1-0.14)
Tvin = Tverre # Température initiale du vin
vin = Cavite( V, rho_v, cp_vin,[ [coque,"G",S_B_i]] , Tvin)

## 4 - Définition du systeme global
# On considère que le temps de service est de 1h soit 3600s
# Le pas de temps vaut Ts=10s
# Notre cavité est le vin
service = 6*3600.0
STL = SystemeThermiqueLineaire(service, 1.0, vin) 

## 5 - Calcul et visualisation des données
# Température extérieure Text
print(f"Température extérieure de {Text:.1f}°C")
coque.definir_signal("D",lambda s,Text=Text : Text/s)
STL.calculer_maintenant()

#!!!!!!!!!!!!!!!!!!!!! FIN DU CALCUL LINÉAIRE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

## Propriétés physiques

# la résistance thermique équivalente est en [m]/[W/K/m] = K/[W/m²]
# (température divisée par densité surfacique de flux)
# np.log(r2/r1)*(r1+r2)/2 / lambd ~ (r2-r1)/lambd

def resist_thq_cyl(couche, verbose=False):
    Req = (couche.Rmax-couche.Rmin)/couche.k
    if verbose :
        print(couche)
        print(f"Résistance thermique ~ {1e6*Req:.2f} K.mm²/W")
    return Req

# Récupérer les valeurs du cas linéaire 
eta_g = eta_vin #Convection au niveau du vin
eta_d = eta_air #Convection au niveau de l'air

#Bouteille

# Récupérer les valeurs du cas linéaire
lamb_verre = verre.k    # 1.05
rho_verre  = verre.rho  # 2500
cp_verre   = verre.Cp   # 720
r_verre    = verre.Rmax # 39.5e-3

r_th_bout = resist_thq_cyl(verre, verbose=True)

#Néoprène

lamb_neo = revet1.k   # 0.192
rho_neo  = revet1.rho # 1230
cp_neo   = revet1.Cp  # 2176

# def des r

r_neo1   = revet1.Rmax # 40e-3
r_plast1 = plast1.Rmax # 40.5e-3
r_MCP    =    mcp.Rmax # 46.5e-3
r_plast2 = plast2.Rmax # 47e-3
r_neo2   = revet2.Rmax # 47.5e-3

#Plastique

lamb_plast = plast1.k   # 0.26
rho_plast  = plast1.rho # 1200
cp_plast   = plast1.Cp  # 1500

#MCP :

from Hprime_Groupe5 import (h_pr_v, h_pr_r, k, T_l_v, T_r_v, T_l_r, T_r_r, \
                          T_min_r, T_max_r, T_min_v, T_max_v, rho_s_v, \
                          rho_l_v, rho_s_r, rho_l_r, cp_MCP)
# Données du RT15 pour le rosé (indice _r)
#et du RT18HC pour le vin rouge (indice _v)

print(f"*** MCP linéaire ***\n{mcp}")
print(f"*** MCP non-linéaire ***\nk = {k:.2f} W/K/m, " + \
      f"rho = {rho_s_v:.2f} kg/m³, Cp = {cp_MCP:.2f} J/K/kg")

# Température extérieure constante et égale à Text
T_air = Text # 25.0

#Discrétisation

Nx = 20
e  = (mcp.Rmax - mcp.Rmin)
dr = e/Nx

#Résistance thermique

r_th_neo1   = resist_thq_cyl(revet1)
r_th_neo2   = resist_thq_cyl(revet2)
r_th_plast1 = resist_thq_cyl(plast1)
r_th_plast2 = resist_thq_cyl(plast2)

Reqg = r_th_bout + r_th_neo1 + r_th_plast1
print(f"Résistance thermique équivalente à gauche ~ {1e6*Reqg:.2f} K.mm²/W")
Reqd = r_th_neo2 + r_th_plast2
print(f"Résistance thermique équivalente à droite ~ {1e6*Reqd:.2f} K.mm²/W")

#Récupération de h'

def alpha(T):
    #Modèle non linéaire
    return k/h_pr_v(T)

def alpha2(T):
    #Modèle linéaire
    return k/(rho_s_v*cp_MCP)*np.ones_like(T)

#Mise sous forme de problème de Cauchy

Vr = np.linspace(mcp.Rmin, mcp.Rmax, Nx+1)[1:-1]

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
#def c(R):

Tinit = T0 * np.ones(Nx) # Récupérer la température initiale linéaire
Tinit[0] = Tvin          # Idem

def FNonLin(Y,t):
    Y = np.array(Y) # Sécurité
    Yprime = np.zeros_like(Y)
    # Première ligne i=0 # Cavité de vin
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

def FLin(Y,t):
    Y = np.array(Y) # Sécurité
    Yprime = np.zeros_like(Y)
    # Première ligne i=0 # Cavité de vin
    Tgstar = 4*dg*Y[1] - dg*Y[2] + gg*Y[0] # Formulaire page 4
    Yprime[0] = 1/(rho_v*cp_vin*V)*S_g*eta_eqg*(Tgstar-Y[0])
    # Deuxième ligne i=1 # Premier nœud du MCP
    Yprime[1] = alpha2(Y[1])*(Ar[0]*Tgstar+b*Y[1]+Cr[0]*Y[2])
    #Lignes de i=2 à i=Nx-2
    for n in range(2,Nx-1):
        Yprime[n] = alpha2(Y[n])*(Ar[n-1]*Y[n-1]+b*Y[n]+Cr[n-1]*Y[n+1])
    #Dernière ligne i=Nx-1
    Tdstar = -dd*Y[-2] + 4*dd*Y[-1] + gd*T_air # Formulaire page 4
    Yprime[-1] = alpha2(Y[-1])*(Ar[-1]*Y[-2]+b*Y[-1]+Cr[-1]*Tdstar)
    return Yprime

# Mémorisation des résultats

soluLin=odeint(FLin,Tinit,Vt)

soluNonLin=odeint(FNonLin,Tinit,Vt)

## 6 - Visualisation des résultats
plt.figure("Résultats de la simulation",figsize=(16,7))
ax_T = plt.subplot(1,2,1)
plt.subplots_adjust(left=0.05, right=0.99, bottom=0.06, top=0.94, wspace=0.3)
# Échelles des instants en minutes
tmn_lin = STL.timeValues/60.0
tmn_NL = Vt/60.0
ax_T1 = plt.subplot(1,2,2)
### Températures ###
for ax in (ax_T,ax_T1) :
    ax.set_title("Températures")
    ax.set_xlabel("Instant t [minutes]")
# Température du vin
T_vin = STL.T_cavites[0]
ax_T.plot(tmn_lin, T_vin, "-", color=(0,0.5,0.5), linewidth=2.0, \
          label = "T° vin SysTherLin")
ax_T.plot(tmn_NL,soluNonLin[:,0],label='T° Vin Non linéaire')
ax_T.plot(tmn_NL,soluLin[:,0],label='T° Vin Linéaire')

ax_T1.plot(tmn_lin, T_vin, ":", color=(0,0.5,0.5))
# La faire apparaître sur l'autre graphique

#------------------------------------------------------------------------------------

#Température du revetement1

_,T_revet1,_= coque.T_phi(77.75e-3/2)
ax_T1.plot(tmn_lin, T_revet1, "-", color="c", linewidth=2.0, label = "T° revet1")

#Température du verre 
_,T_verre,_= coque.T_phi(76.75e-3/2)
ax_T1.plot(tmn_lin, T_verre, "-", color="g", linewidth=2.0, label = "T° verre")

#Température du plast1 
_,T_plast1,_= coque.T_phi(78.25e-3/2)
ax_T1.plot(tmn_lin, T_plast1, "-", color="r", linewidth=2.0, label = "T° plast1")

#Température du mcp 
_,T_mcp,_= coque.T_phi(80.75e-3/2)
ax_T1.plot(tmn_lin, T_mcp, "-", color="b", linewidth=2.0, label = "T° mcp")

#Température du plast2 
_,T_plast2,_= coque.T_phi(83.25e-3/2)
ax_T1.plot(tmn_lin, T_plast2, "-", color="y", linewidth=2.0, label = "T° plast2")

#Température du revetement2 
_,T_revet2,_= coque.T_phi(89.25/2)
ax_T1.plot(tmn_lin, T_revet2, "-", color="m", linewidth=2.0, label = "T° revet2")

# Finalisation du tracé"
ax_T.set_ylabel("Températures [°C]")
ax_T.grid() ; ax_T.legend(loc="best", fontsize=10)
ax_T1.set_ylabel("Températures [°C]")
ax_T1.grid() ; ax_T1.legend(loc="best", fontsize=10)


plt.show()
