# Version 1.24 - 2020, October, 29
# Project : SysTherLin (Systèmes thermiques linéaires)
# Author : Eric Ducasse
# License : CC-BY-NC
# Institution : ENSAM / I2M
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
########################################################################
## EXEMPLE N°1 : AQUARIUM AVEC UN SYSTÈME DE CHAUFFAGE #################
########################################################################
import numpy as np
import matplotlib.pyplot as plt
# Pour aller chercher les modules de SysTherLin :
import os,sys
rel_sys_ther_lin_path = "./SysTherLin" # Chemin relatif
abs_sys_ther_lin_path = os.path.abspath(rel_sys_ther_lin_path)
sys.path.append(abs_sys_ther_lin_path)
# ---
from Couches_conductrices import CoucheConductrice,Multicouche
from Systemes_thermiques_lineaires import (Cavite,\
                                           SystemeThermiqueLineaire)
## 1 - Définition des couches conductrices
T0 = 15.0 # Température initiale commune à tous les éléments
inox = CoucheConductrice(16.5,8000.0,500.0,3.0e-3,T0)
verre_socle =  CoucheConductrice(1.0,2800.0,1000.0,1.0e-3,T0)
verre_coque =  CoucheConductrice(1.0,2800.0,1000.0,8.0e-3,T0)
## 2 - Définition des multicouches avec conditions limites
# 2.1 Socle avec système de chauffage en dessous
socle = Multicouche([inox,verre_socle])
socle.definir_CL("G","Neumann") # Côté chauffage
socle.definir_CL("D","Convection",200.0) # Côté eau (cavité)
# 2.2 Toutes les autres parois
coque = Multicouche([verre_coque])
coque.definir_CL("G","Convection",200.0) # Côté eau (cavité)
coque.definir_CL("D","Convection",10.0) # Côté air extérieur
## 3 - Définition de la cavité
cavite = Cavite(0.2, 1000.0, 4200.0, \
                [(socle,"D",0.4),(coque,"G",1.8)], T0)
## 4 - Définition du systeme global
jour = 3600.*24
STL = SystemeThermiqueLineaire(4*jour,10.0,cavite)
## 5 - Calcul et visualisation des données
# 5.1 Définition du signal de chauffage (domaine temporel)
def chauf(t) : # Fonction vectorisée
    h = (t/3600.0)%24.0  # Heure de la journée
    return ((h>23)|(h<4))*500.0 # 500 W/m² s'il est allumé
instants = STL.timeValues
idx_deb = (instants>=-1e-10).argmax()
instants_positifs = instants[idx_deb:]
socle.definir_signal("G",chauf(instants_positifs))
# Tous les tracés sur la même figure
plt.figure("Graphiques", figsize=(12,8) )
ax_input, ax_output = plt.subplot(2,1,1),plt.subplot(2,1,2)
plt.subplots_adjust(left=0.07, right=0.93, bottom=0.07, top=0.93, \
                    wspace=0.4, hspace=0.3)
ax_input.set_title("Signaux d'entrée")
t_en_jours = instants / jour
ax_input.plot(t_en_jours, socle.signal("G"), ".r", markersize=2)
ax_input.set_ylabel("Puissance fournie [W/m²]", color=(0.6,0,0))
ax_input.grid()
# 5.2 Définition de la température extérieure dans le domaine de Laplace
def T_ext(s,Tinit=T0) :
    # TL de t : Tinit + m*( 1-cos( 2*pi*t/j - a ) )
    m,a,j,dp = 5.0,0.15,3600.0*24,2.0*np.pi
    numer, denom = dp*np.sin(a)+j*s*np.cos(a), dp**2+(j*s)**2
    return (Tinit+m)/s - m*j*numer/denom
coque.definir_signal("D",T_ext)
# Tracé
ax_input2 = ax_input.twinx()
ax_input2.plot(t_en_jours, coque.signal("D"), ".b", markersize=2)
ax_input2.set_ylabel("Température extérieure [°C]",color=(0,0,0.8))
# 5.3 Calcul de la solution
STL.calculer_maintenant()
## 6 - Visualisation des résultats
# Température au coeur de l'inox du socle
_,T_inox,_ = socle.T_phi(1.5e-3) # milieu de l'inox
ax_output.plot(t_en_jours, T_inox, "-", color=(0.8,0,0), \
               linewidth=1.2, label = "T° inox")
# Température des parois de verre
_,T_verre_int,_ = coque.T_phi(coque.X[0]) # intérieur
_,T_verre_ext,_ = coque.T_phi(coque.X[-1]) # extérieur
ax_output.plot(t_en_jours, T_verre_int, "-", color=(0,0,1), \
               linewidth=1.2, label = "T° v.int.")
ax_output.plot(t_en_jours, T_verre_ext, "-", color=(0,0.5,0), \
               linewidth=1.2, label = "T° v.ext.")
# Température de l'eau dans la cavité
T_cav = STL.T_cavites[0]
ax_output.set_title("Résultats de la simulation")
ax_output.set_xlabel("Instant $t$ [jour]")
ax_output.plot(t_en_jours, T_cav, "-", color=(0,0.4,0.8), \
               linewidth=2.0, label = "T° eau")
# Finalisation du tracé
ax_output.set_ylabel("Températures [°C]")
ax_output.grid()
ax_output.legend(loc="best", fontsize=10)
plt.show()
