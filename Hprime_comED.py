from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# données numériques
e = 1.0e-2 # en m
# données du RT15 pour le rosé (indice _r) et du RT18HC pour le vin rouge (indice _v)
k = 2.0e-1 # en W/m/K
T_l_v, T_r_v = 10.0, 17.0 # zone de mélange en °C
T_l_r, T_r_r = 17.0, 19.0 # zone de mélange en °C
T_min_r, T_max_r = 7.0, 22.0 # # températures min et max des données en °C
T_min_v, T_max_v = 10.0, 25.0 # # températures min et max des données en °C
rho_s_v = 880.0 # phase solide  en kg/m^3
rho_l_v = 770.0 # phase liquide en kg/m^3
rho_s_r = 880.0 # phase solide  en kg/m^3
rho_l_r = 770.0 # phase liquide en kg/m^3
# températures aux frontières
T_g, T_d = 10.0, 25.0 # en °C

cp_MCP = 2000

# discrétisation spatiale
n_x = 25 # nombre d'intervalles
delta_x = e / n_x # pas de la discrétisation
X = np.linspace(0, e, n_x+1) # n_x+1 points pour n_x intervalles
X_internes = X[1:-1] # suppression des abscisses extrêmes -> abscisses à l'intérieur du matériau
X_cm = 100.0 * X # abscisses en cm

# discrétisation temporelle
t_min, t_max = 0.0, 10 # en heures
n_t = 501
instants_heures = np.linspace(t_min, t_max, n_t)
instants_secondes = 3600.0 * instants_heures

# températures initiales dans le matériau
def T_init(x, Tg=T_g, Td=T_d, e=e):
    return Tg + x * (Td - Tg + 100.0 * (1.0 - x/e)) / e

temperatures_initiales = T_init(X_internes)

# températures associées aux données graphiques
T_r = np.arange(T_min_r,T_max_r+0.1) # de T_min à T_max incluse
T_v = np.arange(T_min_v,T_max_v+0.1) # de T_min à T_max incluse


# masse volumique en fonction de la température
# approximation affine dans la zone de mélange

rho_v = rho_s_v + (rho_s_v - rho_l_v) * (np.clip(T_v,T_l_v,T_r_v) - T_l_v) / (T_l_v - T_r_v)
rho_r = rho_s_r + (rho_s_r - rho_l_r) * (np.clip(T_r,T_l_r,T_r_r) - T_l_r) / (T_l_r - T_r_r)

# Delta_hm_heat, Delta_hm_cold en kJ/kg
Delta_hm_heat_r = np.array([7, 8,13,14,14,17,16,16,17,18,19,7,3,2,2,3])
Delta_hm_cold_r = np.array([9,10,12,14,13,15,16,15,16,20,14,6,2,2,2,2])
Delta_hm_heat_v = np.array([2,2,2,2,3,5,10, 56,160,9,1,2,2,2,2,2])
Delta_hm_cold_v = np.array([2,2,2,3,4,7,17,125, 89,1,1,1,1,1,2,2])
# Delta_h_moy en J/m^3
Delta_hm_moy_r = 0.5e3 * (Delta_hm_heat_r + Delta_hm_cold_r) # valeurs moyennes
Delta_hm_moy_v = 0.5e3 * (Delta_hm_heat_v + Delta_hm_cold_v) # valeurs moyennes
Delta_h_moy_v = rho_v * Delta_hm_moy_v
Delta_h_moy_r = rho_r * Delta_hm_moy_r


# détermination des valeurs de h' par interpolation linéaire
h_prime_v = interp1d(T_v, Delta_h_moy_v)
def h_pr_v(T=T_v, Tmin=T_min_v, Tmax=T_max_v) :
  return h_prime_v(np.clip(T,Tmin,Tmax))
h_prime_r = interp1d(T_r, Delta_h_moy_r)
def h_pr_r(T=T_r, Tmin=T_min_r, Tmax=T_max_r) :
  return h_prime_r(np.clip(T,Tmin,Tmax))
#!!!!!!!!!!!!!!! Faire un bloc-test pour les tracés :
if __name__ == "__main__" :
    # Tracé de h' en fonction de T
    tab_T = np.linspace(0, 40, 1000) # Prendre plus de points
    tab_h_pr_v = h_pr_v(tab_T)
    tab_h_pr_r = h_pr_r(tab_T)
    plt.figure("h'")
    plt.plot(tab_T, tab_h_pr_v, 'r-', label='Vin rouge')
    plt.plot(T_v, Delta_h_moy_v, 'og', label='RT18HC')
    plt.xlabel(r"Température $T$ [°C]") # Mieux préciser
    plt.ylabel(r"h'")
    plt.tight_layout()
    plt.plot(tab_T, tab_h_pr_r, 'm-', label='Rosé')
    plt.plot(T_r, Delta_h_moy_r, 'db', label='RT15')
    plt.legend()
    plt.grid()
    plt.show()
