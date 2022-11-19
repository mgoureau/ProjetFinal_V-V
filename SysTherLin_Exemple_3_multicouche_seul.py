# Version 1.24 - 2020, October, 29
# Project : SysTherLin (Systèmes thermiques linéaires)
# Author : Eric Ducasse
# License : CC-BY-NC
# Institution : ENSAM / I2M
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
########################################################################
#####    EXEMPLE ÉlÉMENTAIRE D'UN BICOUCHE    ##########################
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
# 3mm d'épaisseur et température initiale de 20°C
inox = CoucheConductrice(16.5,8000.0,500.0,3.0e-3,20.0)
# 10cm d'épaisseur et température initiale de 20°C
verre =  CoucheConductrice(5.0,2800.0,1000.0,0.1,20.0) 
## 2 - Définition du multicouche avec conditions limites
bicouche = Multicouche([inox,verre])
bicouche.definir_CL("G","Neumann") # Côté chauffage
bicouche.definir_CL("D","Convection",10.0) # Côté air extérieur
## ( 3 - Définition de la cavité )
# Sans objet ici
## 4 - Définition du systeme global
jour = 3600.*24
STL = SystemeThermiqueLineaire(4.2*jour,10.0,bicouche)
## 5 - Calcul et visualisation des données
# 5.1 Définition du signal de chauffage (domaine temporel)
def chauf(t) :
    h = (t/3600.0)%24.0
    return ((h>23)|(h<5))*200.0 # 200 W/m² s'il est allumé
# On récupère tous les instants de calcul
instants = STL.timeValues
# On ne sélectionne que les instants positifs (élimination des instants
#   du début)
idx_deb = (instants>=-1e-10).argmax()
instants_positifs = instants[idx_deb:]
# On définit le vecteur représentant les valeurs du signal échantillonné
valeurs_chauffage = chauf(instants_positifs)
# Lorsque le deuxème argument de la méthode definir_signal est un
# vecteur, cela signifie implicitement qu'il s'agit des valeurs du
# signal échantillonné
bicouche.definir_signal("G",valeurs_chauffage)
# 5.2 Définition de la température extérieure (domaine de Laplace)
def TL_T_ext(s) :
    amplitude = 10
    w = 2*np.pi/(24*3600) # Période de 24h
    tau = 10*3600 # Retard pour que le maximum soit à 16h
                  #                    et le minimum à 4h
    return 15/s + amplitude*w*np.exp(-tau*s)/(s**2+w**2)
bicouche.definir_signal("D",TL_T_ext)
# bicouche.signal("D") permettra de récupérer les valeurs temporelles
# 5.3 tracé des courbes
t_en_jours = instants / jour # Les instants d'échantillonnage sont
                             # communs à tous les signaux
plt.figure("Graphiques", figsize=(12,8) )
# Tous les tracés sur la même figure
ax_input, ax_output = plt.subplot(2,1,1),plt.subplot(2,1,2)
plt.subplots_adjust(left=0.07, right=0.93, bottom=0.07, top=0.93, \
                    wspace=0.4, hspace=0.3)
ax_input.set_title("Signaux d'entrée", family="Arial", size=14 )
ax_input.set_xlabel("Instant t [jour]")
ax_input.plot(t_en_jours, bicouche.signal("G"), ".r", markersize=2)
ax_input.set_ylabel("Puissance fournie [W/m²]", color=(0.6,0,0))
ax_input.grid()
ax_input2 = ax_input.twinx() # 2ème système d'axes à droite
ax_input2.plot(t_en_jours, bicouche.signal("D"), ".b", markersize=2)
ax_input2.set_ylabel("Température extérieure [°C]",color=(0,0,0.8))
# 5.3 Calcul de la solution
STL.calculer_maintenant()
## 6 - Visualisation des résultats
ax_output.set_title("Résultats de la simulation", family="Arial", \
                    size=14 )
# Température au coeur de l'inox du socle
z_mil_inox = 0.5*inox.e # position du milieu de l'inox
             # aussi 0.5*(bicouche.X[0]+bicouche.X[1])
_,T_inox,_ = bicouche.T_phi(z_mil_inox)
ax_output.plot(t_en_jours,T_inox,"-",color=(0.8,0,0),linewidth=2.0,\
               label = "T° inox, $x={:.1f}$ mm".format(1e3*z_mil_inox))
# Température du verre du côté extérieur
Vx = np.array([0.8*bicouche.X[1]+0.2*bicouche.X[2], \
               0.5*bicouche.X[1]+0.5*bicouche.X[2], bicouche.X[2]])
for x in Vx :
    _,T_verre,_ = bicouche.T_phi(x)
    ax_output.plot(t_en_jours, T_verre, "-", linewidth=2.0,\
                label = "T° verre, $x={:.1f}$ mm".format(1e3*x))
# Finalisation du tracé
ax_output.set_ylabel("Températures [°C]")
ax_output.grid()
ax_output.legend(loc="best",fontsize=10)
# Affichage à la fin
plt.show()
