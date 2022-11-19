# Version 1.24 - 2020, October, 29
# Project : SysTherLin (Systèmes thermiques linéaires)
# Author : Eric Ducasse
# License : CC-BY-NC
# Institution : ENSAM / I2M
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Modélisation de multicouches linéaires 1D plan, ou à symétrie
# cylindrique ou sphérique. Les solutions exactes sont calculées dans
# le domaine de Laplace, avant de revenir dans le domaine temporel par
# transformée de Laplace inverse numérique.
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import numpy as np
from numpy.linalg import solve
from numpy.fft import rfft,irfft
from scipy.special import erf,ive,kve
#==================== CLASSE COUCHE CONDUCTRICE ========================
class CoucheConductrice :
    """Couche conductrice thermique uniforme 1D."""
    def __init__(self,k,rho,Cp,e,Tinit=0.0) :
        """ k : conductivité thermique [W/m/K]
            rho : masse volumique [kg/m^3]
            Cp : capacité calorifique massique [J/K/kg]
            e : épaisseur de la couche [m].
            Tinit : écart initial de température (supposée uniforme) 
                    par rapport à la température de référence."""
        self.__k = k
        self.__r = rho
        self.__c = Cp
        self.__updateUnSurAlpha()
        self.__e = e
        self.__Ti = Tinit
    #-------------------------------------------------------------------
    def __updateUnSurAlpha(self) :
        """Stockage de 1/alpha = rho*Cp/k [m^2*s]."""
        self.__1sa = self.__r*self.__c/self.__k
    # attributs en lecture seule :
    @property
    def k(self): return self.__k
    @property
    def rho(self): return self.__r
    @property
    def Cp(self): return self.__c
    @property
    def un_sur_alpha(self): return self.__1sa
    @property
    def e(self): return self.__e
    @property
    def tau(self):
        """Constante de temps [s] : e^2/(2*alpha)"""
        return 0.5*self.un_sur_alpha*self.e**2
    @property
    def delta_T_initial(self):
        """Écart initial de température par rapport à la
           température de référence"""
        return self.__Ti
    @property
    def Tinit(self) : return self.__Ti
    #-------------------------------------------------------------------
    # pour l'affichage par print :
    def __str__(self) :
        msg = "Couche conductrice de paramètres :"
        for nom,val,u in [["Conductivité",self.k,"W/K/m"],
                    ["Masse volumique",self.rho,"kg/m^3"],
                    ["Capacité calorique",self.Cp,"J/K/kg"],
                    ["Épaisseur",1000*self.e,"mm"],
                    ["Constante de temps",self.tau,"s"]] :
            msg += "\n\t{} : {:.2f} {}".format(nom,val,u)
        msg += "\n\tTempérature initiale : " + \
               "{:.2f} °C".format(self.Tinit)
        return msg 
    #-------------------------------------------------------------------
    # Matrice qui permet de calculer T et phi
    def P(self,s,x=None) :
        """s est un vecteur de valeurs complexes. Pour un s donné,
           matrice P telle que
                 (T(x),phi(x)) = P(x).(a,b) + (Tinit/s,0).
           a correspond à la solution en exp(-sqrt(s/alpha)x).
           b correspond à la solution en exp(-sqrt(s/alpha)(e-x))."""
        if x is None : x = self.e # côté droit
        s = np.array(s) # au cas où s soit une valeur ou une liste
        M = np.ndarray( list(s.shape)+[2,2], dtype = complex )
        R = np.sqrt(s*self.un_sur_alpha)
        X,EmX = R*x,R*(self.e-x)
        Eplus,Emoins = np.exp(-X),np.exp(-EmX)
        M[:,0,0],M[:,0,1] = Eplus,Emoins
        M[:,1,0],M[:,1,1] = R*Eplus,-R*Emoins
        M[:,1,:] *= self.k
        return M
#============== CLASSE COUCHE CONDUCTRICE CYLINDRIQUE ==================
class CoucheConductriceCylindrique(CoucheConductrice) :
    """Couche conductrice thermique uniforme 1D, à symétrie cylindrique.
    """
    def __init__(self,k,rho,Cp,Rmin,Rmax,Tinit=0.0) :
        """ k : conductivité thermique [W/m/K]
            rho : masse volumique [kg/m^3]
            Cp : capacité calorifique massique [J/K/kg]
            Rmin : rayon intérieur de la couche [m]
            Rmax : rayon extérieur de la couche [m].
            Tinit : écart initial de température (supposée uniforme) 
                    par rapport à la température de référence."""
        CoucheConductrice.__init__(self,k,rho,Cp,Rmax-Rmin,Tinit)
        self.__r_min = Rmin
        self.__r_max = Rmax
        self.__r_mid = 0.5*(Rmin+Rmax) 
    #-------------------------------------------------------------------
    @property
    def Rmin(self) : return self.__r_min
    @property
    def Rmax(self) : return self.__r_max 
    @property
    def tau(self):
        """Constante de temps [s] : (formule semi-empirique)"""
        tau0 = 0.25*self.un_sur_alpha*self.e**2
        rho = self.Rmin/self.Rmax
        g,c = 0.30440, 0.68012
        l0,l1,lr = np.log(g),np.log(g+1),np.log(g+rho)
        f = c*( 1-rho + (l1-lr)/(l0-l1) )
        return tau0*(1+rho+f)
    #-------------------------------------------------------------------
    # pour l'affichage par print :
    def __str__(self) :
        msg = "Couche conductrice cylindrique de paramètres :"
        for nom,val,u in [["Conductivité",self.k,"W/K/m"],
                    ["Masse volumique",self.rho,"kg/m^3"],
                    ["Capacité calorique",self.Cp,"J/K/kg"],
                    ["Épaisseur",1000*self.e,"mm"],
                    ["Rayon intérieur",1000*self.Rmin,"mm"],
                    ["Rayon extérieur",1000*self.Rmax,"mm"],
                    ["Constante de temps",self.tau,"s"]] :
            msg += "\n\t{} : {:.2f} {}".format(nom,val,u)
        msg += "\n\tTempérature initiale : " + \
               "{:.2f} °C".format(self.Tinit)
        return msg 
    #-------------------------------------------------------------------                          
    # Matrice qui permet de calculer T et phi
    def P(self,s,r=None) :
        """s est un vecteur de valeurs complexes. Pour un s donné,
           matrice P telle que
                 (T(x),phi(x)) = P(x).(a,b) + (Tinit/s,0).
           a correspond à la solution en K_n (Bessel mod. 2nde espèce).
           b correspond à la solution en I_n (Bessel mod. 1ère espèce).
           """
        #kve(v, z) = kv(v, z) * exp(z)
        #ive(v, z) = iv(v, z) * exp(-abs(z.real))
        if r is None : r = self.Rmax # côté droit
        s = np.array(s) # au cas où s soit une valeur ou une liste
        M = np.ndarray( list(s.shape)+[2,2], dtype = complex )
        B = np.sqrt(s*self.un_sur_alpha)
        if r <= 0 : # Divergence sauf si a = 0
            Z = np.zeros_like(s)
            M[:,0,0],M[:,0,1] = Z,np.exp(-(B*self.__r_mid).real)
            M[:,1,0],M[:,1,1] = Z,Z # Flux nul en r=0 (symétrie)
            return M
        Br,Bdr = B*r,B*(r-self.__r_mid)
        Eplus,Emoins = np.exp(-Bdr),np.exp(Bdr.real)
        M[:,0,0],M[:,0,1] = Eplus*kve(0,Br),Emoins*ive(0,Br)
        M[:,1,0],M[:,1,1] = Eplus*kve(1,Br),-Emoins*ive(1,Br)
        M[:,1,:] = np.einsum("i,ij->ij",self.k*B,M[:,1,:])
        return M
#============== CLASSE COUCHE CONDUCTRICE SPHÉRIQUE ====================
class CoucheConductriceSpherique(CoucheConductrice) :
    """Couche conductrice thermique uniforme 1D, à symétrie sphérique.
    """
    def __init__(self,k,rho,Cp,Rmin,Rmax,Tinit=0.0) :
        """ k : conductivité thermique [W/m/K]
            rho : masse volumique [kg/m^3]
            Cp : capacité calorifique massique [J/K/kg]
            Rmin : rayon intérieur de la couche [m]
            Rmax : rayon extérieur de la couche [m].
            Tinit : écart initial de température (supposée uniforme) 
                    par rapport à la température de référence."""
        CoucheConductrice.__init__(self,k,rho,Cp,Rmax-Rmin,Tinit)
        self.__r_min = Rmin
        self.__r_max = Rmax
        self.__r_mid = 0.5*(Rmin+Rmax) 
    #-------------------------------------------------------------------
    @property
    def Rmin(self) : return self.__r_min
    @property
    def Rmax(self) : return self.__r_max
    @property
    def tau(self):
        """Constante de temps [s] : alpha/(6*e^2) * (1+2*Rmin/Rmax)"""
        tau0 = self.un_sur_alpha*self.e**2/6
        rho = self.Rmin/self.Rmax
        return tau0*( 1 + 2*rho ) 
    #-------------------------------------------------------------------
    # pour l'affichage par print :
    def __str__(self) :
        msg = "Couche conductrice sphérique de paramètres :"
        for nom,val,u in [["Conductivité",self.k,"W/K/m"],
                    ["Masse volumique",self.rho,"kg/m^3"],
                    ["Capacité calorique",self.Cp,"J/K/kg"],
                    ["Épaisseur",1000*self.e,"mm"],
                    ["Rayon intérieur",1000*self.Rmin,"mm"],
                    ["Rayon extérieur",1000*self.Rmax,"mm"],
                    ["Constante de temps",self.tau,"s"]] :
            msg += "\n\t{} : {:.2f} {}".format(nom,val,u)
        msg += "\n\tTempérature initiale : " + \
               "{:.2f} °C".format(self.Tinit)
        return msg 
    #-------------------------------------------------------------------                          
    # Matrice qui permet de calculer T et phi
    def P(self,s,r=None) :
        """s est un vecteur de valeurs complexes. Pour un s donné,
           matrice P telle que
                 (T(x),phi(x)) = P(x).(a,b) + (Tinit/s,0).
           a correspond à la solution en exp(-sqrt(s/alpha)(r-Rmin)).
           b correspond à la solution en exp(-sqrt(s/alpha)(Rmax-r)).
           """
        if r is None : r = self.Rmax # côté droit
        s = np.array(s) # au cas où s soit une valeur ou une liste
        M = np.ndarray( list(s.shape)+[2,2], dtype = complex )
        B = np.sqrt(s*self.un_sur_alpha)
        if r <= 0 : # Divergence sauf si a = -b*exp(-B*Rmax)
            L0,Z = B*np.exp(-B*self.Rmax),np.zeros_like(s)
            M[:,0,0],M[:,0,1] = -B,L0
            M[:,1,0],M[:,1,1] = Z,Z # Flux nul en r=0 (symétrie)
            return M
        Br = B*r
        Eplus,Emoins = np.exp(B*(self.Rmin-r)),np.exp(B*(r-self.Rmax))
        M[:,0,0],M[:,0,1] = Eplus,Emoins
        M[:,0,:] /= r
        M[:,1,0],M[:,1,1] = Eplus*(1+Br),Emoins*(1-Br)
        M[:,1,:] *= self.k/r**2
        return M        
##################### TESTS ÉLÉMENTAIRES ###############################
if __name__ == "__main__" :
    brique = CoucheConductrice(0.84,1800.0,840.0,0.220)
    print(brique)
    couche_cyl = CoucheConductriceCylindrique(\
                                        0.84,1800.0,840.0,0.02,0.025)
    print(couche_cyl)
    print("couche_cyl.P([1.0+2.0j],0.021) :")
    print(couche_cyl.P([1.0+2.0j],0.021)[0])
    couche_sph = CoucheConductriceSpherique(\
                                        0.84,1800.0,840.0,0.02,0.025)
    print(couche_sph)
    print("couche_sph.P([1.0+2.0j],0.021) :")
    print(couche_sph.P([1.0+2.0j],0.021)[0])
#======================= CLASSE MULTICOUCHE ============================
class Multicouche :
    """Multicouche ne contenant que des couches conductrices."""
    def __init__(self,couches,x_min=None) :
        if isinstance(couches,CoucheConductrice) :
            couches = [couches]
        self.__n = len(couches)  # nombre de couches conductrices
        self.__couches = couches # Liste des couches conductrices
        self.__type = None
        if isinstance(couches[0],CoucheConductriceCylindrique) :
            self.__cyl_or_sph = True
            self.__type = "cyl"
        elif isinstance(couches[0],CoucheConductriceSpherique) :
            self.__cyl_or_sph = True
            self.__type = "sph"
        else : # couches planes
            self.__cyl_or_sph = False
            self.__type = "pla"
        if x_min is None :
            if self.__cyl_or_sph :
                x_min = couches[0].Rmin
            else :
                x_min = 0.0
        elif self.__cyl_or_sph :
            if abs(x_min-couches[0].Rmin) > 1e-12*couches[0].e :
                msg = ("Constructeur de Multicouche :: attention : "+\
                       "x_min [{:.3e}] n'est pas voisin de Rmin "+\
                       "[{:.3e}] !").format(x_min,couches[0].Rmin)
                print(msg)
        if self.__cyl_or_sph and abs(x_min)<1e-14*couches[0].e :
            # Côté gauche confondu avec l'axe/le centre de symétrie
            self.__centre = True
        else : self.__centre = False
        X,x = [x_min],x_min
        for nc,couche in enumerate(couches,1) :
            x += couche.e
            if self.__cyl_or_sph and \
               abs(x-couche.Rmax) > 1e-12*couche.e :
                msg = ("Constructeur de Multicouche :: attention : "+\
                       "x [{:.3e}] n'est pas voisin de Rmax "+\
                       "[{:.3e}] dans la couche {} !").format(\
                           x,couche.Rmax,nc)
                print(msg)                
            X.append(x)
        self.__X = np.array(X) # positions des interfaces
        self.__d = None     # durée de la simulation
        self.__Ts = None    # période d'échantillonnage
        self.__nt = None    # nombre de points des signaux
        self.__r = 8        # nombre de points du côté t<0
        self.__s = None     # Vecteur de complexes (domaine de
                            #     Laplace)
        #+++++++++ Matrice du système à résoudre 2n x 2n ++++++++++++
        self.__AB = None    # coefficients (ai,bi)
        #+++++++++ Côté gauche ++++++++++++
        if self.__centre : self.__CLG = "(Symétrie)"
        else : self.__CLG = None   # Condition limite à gauche
        self.__TLFG = None  # Fonction représentant la T.L. du signal
                            # imposé à droite
        self.__VG = None    # Valeurs de la T.L. de la fonction
                            #     imposée à gauche
        self.__sigG = None  # Signal numérique imposé à gauche
        self.__hG = None    # Coefficient de convection à gauche
        self.__TiG = couches[0].Tinit   # Température initiale
        #+++++++++ Côté droit ++++++++++++
        self.__CLD = None   # Condition limite à droite
        self.__TLFD = None  # Fonction représentant la T.L. du signal
                            # imposé à droite
        self.__VD = None    # Valeurs de la T.L. de la fonction
                            #     imposée à droite
        self.__sigD = None  # Signal numérique imposé à droite
        self.__hD = None    # Coefficient de convection à droite 
        self.__TiD = couches[-1].Tinit   # Température initiale
    #-------------------------------------------------------------------
    @property
    def nb(self) :
        """Nombre de couches conductrices."""
        return self.__n
    @property
    def X(self) :
        """Positions des bords et interfaces"""
        return np.array(self.__X.tolist())
    @property
    def Tinit_gauche(self) :
        return self.__TiG
    @property
    def Tinit_droite(self) :
        return self.__TiD
    #-------------------------------------------------------------------
    def definir_temps(self,duree,Ts) :
        """'duree' est la durée de la simulation et 'Ts' la
           période d'échantillonnage."""
        self.__Ts = Ts
        self.__nt = 2*int(np.ceil(0.5*duree/Ts)) # nombre pair de
                    # pas de temps
        self.__nt += self.__r
        self.__d = self.__nt*Ts
        gamma = 11.5 / duree  # partie réelle des valeurs de s
        df = 1.0 / self.__d   # pas de discrétisation en fréquences
        s_values = gamma + 2j*np.pi*df*np.arange(0,self.__nt//2+1,1)
        self.__s = s_values
        self.__update_VH_VG()
        self.__update_AB() # Calcul, si c'est possible de la solution 
    #-------------------------------------------------------------------
    def definir_signal(self,cote,H) :
        """'cote' vaut 'G' (gauche) ou 'D' (droite).
           'H' est la transformée de Laplace (fonction) du signal,
           ou un signal numérique que l'on filtrera avant de calculer
           sa transformée de Laplace Numérique."""
        if cote.lower() in ["g","gauche"] :
            if self.__centre :
                print("Multicouche.definir_signal :: attention :"+\
                      " Inutile de définir un signal sur le "+\
                      "centre de symétrie !")
                self.__TLFG = None
            else :
                if callable(H) : # H est une fonction, transformée de
                                 # Laplace du signal
                    self.__TLFG = H
                else : # Signal numérique partant de t=0
                    self.__sigG = self.__set_signal(H)
        elif cote.lower() in ["d","droite"] :
            if callable(H) : # H est une fonction, transformée de
                             # Laplace du signal
                self.__TLFD = H
            else : # Signal numérique partant de t=0
                self.__sigD = self.__set_signal(H)
        else :
            print('cote "{}" inconnu'.format(cote))
            return
        self.__update_VH_VG() 
    #-------------------------------------------------------------------        
    def __set_signal(self,signal) :
        deb_msg = "Multicouche.definir_signal :: erreur : "
        try :
            sig = np.array(signal)
        except :
            msg = deb_msg+"Échec de la conversion du signal de type "+\
                  "'{}' en ndarray".format(type(signal).__name__)
            raise ValueError(msg)
        nbval = self.__nt-self.__r
        if sig.shape != (nbval,) :
            msg = deb_msg + "Le signal fourni doit être de forme "+\
                              "{} et non pas {} !".format(\
                                  (nbval,),sig.shape)
            raise ValueError(msg)
        return np.append(np.zeros(self.__r),sig)
    #-------------------------------------------------------------------                                                                          
    def __update_VH_VG(self) :
        if self.__s is None :
            self.__VG = None
            self.__VD = None
            self.__sigG = None
            self.__sigD = None
            return
        # Prise en compte des températures initiales ici
        if self.__TLFG is None :
            if self.__sigG is None :
                self.__VG = None
            else :
                signalG = self.__sigG[self.__r:]
                if self.__CLG in ("Dirichlet","Convection") :
                    signalG -= self.__TiG
                self.__VG = self.TLdir(signalG)
        else :
            if self.__CLG in ("Dirichlet","Convection") :
                self.__VG = self.__TLFG(self.__s)-self.__TiG/self.__s
            else :
                self.__VG = self.__TLFG(self.__s)
            self.__sigG = self.TLrec(self.__VG)
        if self.__TLFD is None : 
            if self.__sigD is None :
                self.__VD = None
            else :
                signalD = self.__sigD[self.__r:]
                if self.__CLD in ("Dirichlet","Convection") :
                    signalD -= self.__TiD
                self.__VD = self.TLdir(signalD)
        else :
            if self.__CLD in ("Dirichlet","Convection") :
                self.__VD = self.__TLFD(self.__s)-self.__TiD/self.__s
            else :
                self.__VD = self.__TLFD(self.__s)
            self.__sigD = self.TLrec(self.__VD)
        self.__update_AB() # Calcul, si c'est possible de la solution 
    #-------------------------------------------------------------------
    def definir_CL(self,cote,CL_type,coef_convection=None) :
        if CL_type.lower() in ["d","dirichlet"] :
            CL_type = "Dirichlet"
        elif CL_type.lower() in ["n","neumann"] :
            CL_type = "Neumann"
        elif CL_type.lower() in ["c","conv","convec","convection"] :
            CL_type = "Convection"
            if coef_convection is None :
                print("Coefficient de convection manquant !")
                return
        elif CL_type.lower() in ["s","(s)","sym","(sym)","symetrie",\
                                 "(symetrie)","symétrie","(symétrie)"]:
            CL_type = "(Symétrie)"
        else :
            print("Type '{}' de condition limite non reconnu.".format(\
                CL_type))
            return
        if cote.lower() in ["g","gauche"] :
            if self.__centre and CL_type in ["Dirichlet","Neumann",\
                                                "Convection"] :
                print("Multicouche.definir_CL :: attention :"+\
                      " Inutile de définir la condition limite "+\
                      "au centre : symétrie !")
                CL_type = "(Symétrie)"
            self.__CLG = CL_type
            if CL_type == "Convection" : self.__hG = coef_convection
            else : self.__hG = None
            #print("hG initialisé :",self.__hG)
        elif cote.lower() in ["d","droite"] :
            self.__CLD = CL_type
            if CL_type == "Convection" : self.__hD = coef_convection
            else : self.__hD = None
            #print("hD initialisé :",self.__hD)
        else :
            print('cote "{}" inconnu'.format(cote))
            return
        self.__update_AB() # Calcul, si c'est possible de la solution 
    #-------------------------------------------------------------------
    def __check(self,verbose=False,signals=(True,True)) :        
        """Renvoie un booléen qui indique si le calcul est possible."""
        if verbose : prt = print # verbose : affichage erreurs
        else :
            def prt(*a,**k) : pass
        chk = True
        if self.__s is None :
            prt("Valeurs de la T.L. non définies")
            chk = False
        if signals[0] and not self.__centre and self.__VG is None :
            prt("Signal à gauche non défini.")
            chk = False
        if self.__CLG is None :
            prt("Condition limite à gauche non définie.")
            chk = False
        if signals[1] and self.__VD is None :
            prt("Signal à droite non défini.")
            chk = False
        if self.__CLD is None :
            prt("Condition limite à droite non définie.")
            chk = False
        return chk 
    #-------------------------------------------------------------------
    def tout_est_defini(self,signals=(True,True)) :
        """Vérifie que tout est bien défini."""
        return self.__check(verbose=True,signals=signals) 
    #-------------------------------------------------------------------
    def M_matrix(self,raised_errors=True,verbose=False) :
        if not self.__check(signals=(False,False)) : # Calcul impossible
            msg = "Multicouche.M_matrix :: données insuffisantes pour"+\
                  " calculer la matrice M."""
            if raised_errors : raise ValueError(msg)
            elif verbose : print(msg)
            return None
        dn,s,couches = 2*self.nb,self.__s,self.__couches
        M = np.zeros( list(s.shape)+[dn,dn], dtype=complex)
        # Condition limite à gauche
        cm1 = couches[0]
        P0 = cm1.P(s,self.__X[0])
        if self.__CLG == "(Symétrie)" :
            Z,U = np.zeros_like(s),np.ones_like(s)
            if self.__type == "cyl" :
                M[:,0,:2] = np.array([U,Z]).transpose() # a=0
            elif self.__type == "sph" : # a + exp(-B*e)*b = 0
                M[:,0,:2] = np.array( \
                    [U,np.exp(-cm1.e*np.sqrt(s*cm1.un_sur_alpha))] \
                    ).transpose()
            else : # impossible
                print("Problème ! CL (Symétrie) dans le cas plan")
                return None
        elif self.__CLG == "Dirichlet" :
           M[:,0,:2] = P0[:,0,:]
        elif self.__CLG == "Neumann" :
           M[:,0,:2] = P0[:,1,:]
        elif self.__CLG == "Convection" :
            M[:,0,:2] = P0[:,0,:]+P0[:,1,:]/self.__hG
        else :
            msg = "CLG '{}' impossible !".format(self.__CLG)
            if raised_errors : raise ValueError(msg)
            else :
                print(msg)
                return None           
        # Continuité aux interfaces
        for i,(c,x) in enumerate(zip(couches[1:],self.__X[1:])) :
            di = 2*i
            if self.__cyl_or_sph : P0,P1 = cm1.P(s,x),c.P(s,x)
            else : P0,P1 = cm1.P(s),c.P(s,0)
            M[:,di+1:di+3,di:di+2] = P0
            M[:,di+1:di+3,di+2:di+4] = -P1
            cm1 = c
        # Condition limite à droite
        if self.__cyl_or_sph : P1 = cm1.P(s,self.__X[-1])
        else : P1 = cm1.P(s)
        if self.__CLD == "Dirichlet" :
            M[:,-1,-2:] = P1[:,0,:]
        elif self.__CLD == "Neumann" :
            M[:,-1,-2:] = P1[:,1,:]
        elif self.__CLD == "Convection" :
            M[:,-1,-2:] = P1[:,0,:]-P1[:,1,:]/self.__hD
        else :
            msg = "CLD '{}' impossible !".format(self.__CLD)
            if raised_errors : raise ValueError(msg)
            else :
                print(msg)
                return None           
        return M 
    #-------------------------------------------------------------------        
    def __update_AB(self) :
        """Déterminer les T.L. des températures et des densités
           surfaciques de flux sur les bords."""
        M = self.M_matrix(raised_errors=False)
        if M is None : return
        # print((abs(M[-1,:,:])>1e-8)*1) # Contrôle de la forme de M
        # Second membre
        S = np.zeros( M.shape[:-1], dtype=complex)
        if not self.__check() : # Calcul impossible
            self.__AB = None
            return
        # Côté gauche
        if not self.__centre :
            S[:,0] = self.__VG
        # Interfaces avec prise en compte des températures initiales
        TinitL = self.couches[0].Tinit
        for i,layer in enumerate(self.couches[1:],1) :
            TinitR = layer.Tinit
            S[:,2*i-1] += (TinitR-TinitL)/self.__s
            TinitL = TinitR
        # Côté droit
        S[:,-1] = self.__VD
        # Résolution
        self.__AB = solve(M,S)
        return 
    #-------------------------------------------------------------------
    def set_AB(self,new_AB) :
        shp = self.__s_shape+(2*self.nb,)
        if new_AB.shape == shp :
            self.__AB = new_AB
            return
        else :
            msg = "Multicouche.set_AB :: Erreur : tailles "+\
                  "incompatibles : {} donné pour {} demandé.".format(\
                      new_AB.shape,shp)
            raise ValueError(msg) 
    #-------------------------------------------------------------------        
    def T_phi(self,x) :
        """ Renvoie les vecteurs 'instants','température' et
            'densité surfacique de flux' à la position x (scalaire)."""
        if self.__AB is None :
            print("Impossible de calculer les champs : "+\
                  "coefficients inconnus")
        if x <= self.__X[0] : x = self.__X[0]
        elif x >= self.__X[-1] : x = self.__X[-1]
        xg = self.__X[0]
        for i,(xd,c) in enumerate(zip(self.__X[1:],self.__couches)) :
            if x <= xd :
                if self.__cyl_or_sph : P = c.P(self.__s,x)
                else : P = c.P(self.__s,x-xg)
                V = np.einsum("ijk,ik->ij",P,self.__AB[:,2*i:2*i+2])
                T,phi = V[:,0],V[:,1]
                # T et phi désignent la TL des signaux cherchés
                return self.timeValues, \
                       self.TLrec(T)+c.Tinit,self.TLrec(phi)
            xg = xd 
    #-------------------------------------------------------------------
    @property
    def timeValues(self) :
        """Instants auxquels sont calculées les réponses du système."""
        Ts = self.__Ts
        t0 = -self.__r*Ts   # Instant de début du signal
        return np.array([t0+i*Ts for i in range(self.__nt)]) 
    #-------------------------------------------------------------------
    def signal(self,cote) :
        if cote.lower() in ['g','gauche'] :
            if self.__centre :
                print("Multicouche.signal :: Condition de symétrie !")
                return np.zeros(self.__nt)
            if self.__CLG in ("Dirichlet","Convection") :
                return self.__sigG + self.__TiG
            else :
                return self.__sigG
        elif cote.lower() in ['d','droite'] :
            if self.__CLD in ("Dirichlet","Convection") :
                return self.__sigD + self.__TiD
            else :
                return self.__sigD
        msg = ("Multicouche.signal :: côté '{}' inconnu !".format(\
                cote))
        raise ValueError(msg)
    #-------------------------------------------------------------------
    def TLdir(self,signal) :
        if self.__s is None :
            msg = "Multicouche.TLdir :: Erreur : impossible de "+\
                  "calculer une transformée de Laplace numérique "+\
                  "puisque le temps n'est pas défini. Il faut d'abord "+\
                  "appeler la méthode 'definir_temps'."
            raise ValueError(msg)            
        signal = np.array(signal)
        nb_val = self.__nt-self.__r
        if signal.shape != (nb_val,) :
            Ts = self.__Ts
            msg = "Multicouche.TLdir :: Erreur : le signal est de for"+\
                  ("me {} et non pas {} (signal causal de 0.0 à "+\
                   "{:.5e} s par pas de {:.3e} s").format(signal.shape,\
                    (nb_val,),(nb_val-1)*Ts,Ts)
            raise ValueError(msg)        
        gamma = self.__s[0].real
        signal = np.append(signal,np.zeros(self.__r))
        mt0 = self.__r*self.__Ts
        signal *= np.exp(-gamma*(self.timeValues+mt0))        
        return rfft(signal)*self.__Ts 
    #-------------------------------------------------------------------
    def TLrec(self,U) :
        """Renvoie les valeurs du signal numérique dont la T.L.
           est le vecteur U (pour les valeurs s dans self.__s."""
        if U.shape != self.__s.shape :
            print("Anomalie : U et s n'ont pas la même dimension.")
            return
        # Filtrage passe-bas pour la discontinuité en t=0
        a = 4.0/np.pi
        ar = -self.__r
        b = -np.sqrt(a)
        br = 0.25*np.sqrt(np.pi)*self.__r
        Sr = self.__Ts*self.__s
        Y =  0.5*U*np.exp(Sr*(a*Sr+ar))*(1+erf(b*Sr+br))
        Y[-1] = 0.5*Y[-1].real
        gamma = self.__s[0].real
        mt0 = self.__r*self.__Ts
        return irfft(Y)/self.__Ts*np.exp(gamma*(self.timeValues+mt0))
    #-------------------------------------------------------------------
    def TLsig(self,cote,offset=True) :
        if cote.lower() in ["g","gauche"] :
            if self.__centre :
                print("Multicouche.signal :: Condition de symétrie !")
                return np.zeros_like(self.__s)
            if offset and self.__CLG in ("Dirichlet","Convection") :
                return self.__VG + self.__TiG/self.__s
            else :
                return self.__VG
        elif cote.lower() in ["d","droite"] :
            if offset and self.__CLD in ("Dirichlet","Convection") :
                return self.__VD + self.__TiD/self.__s
            else :
                return self.__VD
        msg = ("Multicouche.TLsig :: Erreur : côté '{}' inconnu"+\
               ".").format(cote)
        raise ValueError(msg)
    #-------------------------------------------------------------------        
    def __str__(self) :
        msg = "Multicouche à {} couche(s) conductrice(s) {}\n"
        if self.__type == "plan" : tp = "plane(s)"
        elif self.__type == "cyl" : tp = "cylindrique(s)"
        elif self.__type == "sph" : tp = "sphérique(s)"
        else : tp = "" # (erreur)
        msg = msg.format(self.__n,tp)
        msg += "\td'épaisseur(s) en millimètres : ["
        for c in self.__couches : msg += "{:.1f},".format(1000*c.e)
        msg = msg[:-1]+"]"
        msg += '\n\tCondition limite à gauche : "{}"'.format(self.__CLG)
        msg += '\n\tCondition limite à droite : "{}"'.format(self.__CLD)
        return msg 
    #-------------------------------------------------------------------
    @property
    def couches(self) : return self.__couches
    @property
    def geometrie(self) :
        if self.__type == "plan" : return "plane"
        if self.__type == "cyl" : return "cylindrique"
        if self.__type == "sph" : return "sphérique"
    @property
    def CLG(self) :
        if self.__centre : return ("(Symétrie)",)
        if self.__CLG == "Convection" :
            return ("Convection",self.__hG,self.__VG)
        elif self.__CLG in ["Dirichlet","Neumann"] :
            return (self.__CLG,self.__VG)
        else :
            return ("Non définie",)
    @property
    def CLD(self) :
        if self.__CLD == "Convection" :
            return ("Convection",self.__hD,self.__VD)
        elif self.__CLG in ["Dirichlet","Neumann"] :
            return (self.__CLD,self.__VD)
        else :
            return ("Non définie",)
    @property
    def s(self) : return self.__s
    @property
    def __s_shape(self) :
        if self.__s is None : return (0,)
        else : return self.__s.shape
    @property
    def ns(self) : return  self.__s_shape[0]
##################### TESTS ÉLÉMENTAIRES ###############################
if __name__ == "__main__" :
    choix = 4
    if choix == 1 : ## Exemple de multicouche plan 
        # Étape 1 - Définition du multicouche
        nb = 4
        petite_brique = CoucheConductrice(0.84,1800.0,840.0,0.220/nb,\
                                          Tinit=20.0)
        C = Multicouche(nb*[petite_brique])
    elif choix == 2 : ## Exemple de multicouche cylindrique
        # Étape 1 - Définition du multicouche
        Rmin = 0 ; Rmax = Rmin+0.220
        nb = 4
        VR = np.linspace(Rmin,Rmax,nb+1)
        couches = []
        for rmin,rmax in zip(VR[:-1],VR[1:]) :
            couches.append(\
              CoucheConductriceCylindrique(0.84,1800.0,840.0,rmin,rmax,\
                                          Tinit=20.0))
        C = Multicouche(couches)
    elif choix == 3 : ## Exemple de multicouche sphérique
        # Étape 1 - Définition du multicouche
        Rmin = 0.5 ; Rmax = Rmin+0.220
        nb = 3
        VR = np.linspace(Rmin,Rmax,nb+1)
        couches = []
        for rmin,rmax in zip(VR[:-1],VR[1:]) :
            couches.append(\
              CoucheConductriceSpherique(0.84,1800.0,840.0,rmin,rmax,\
                                          Tinit=20.0))
        C = Multicouche(couches)
    if choix == 4 : ## Exemple de multicouche plan avec températures
                    ##  initiales différentes : Versions >= 1.2
        brique_chaudeG = CoucheConductrice(0.84,1800.0,840.0,0.05,\
                                          Tinit=54.0)
        brique_froide = CoucheConductrice(0.84,1800.0,840.0,0.14,\
                                          Tinit=10.0)
        brique_chaudeD = CoucheConductrice(0.84,1800.0,840.0,0.03,\
                                          Tinit=54.0)
        C = Multicouche([brique_chaudeG,brique_froide,brique_chaudeD])
    # Étape 2 - Définition de la durée et de la période
    #           d'échantillonnage
    if choix != 4 :
        C.definir_temps(2.0e5,0.5e3)
    else :
        C.definir_temps(3.5e4,30.0)
    # Étape 3 - Définition des conditions aux limites
    # 3a/ À gauche
    if choix != 2 : # pour choix == 2, condition de symétrie
        C.definir_CL("G","Neumann")
    # Transformée de Laplace du signal imposé : ici signal nul
        C.definir_signal("G",np.zeros_like)
    # 3b/ À droite
    if choix != 4 :
        C.definir_CL("D","Dirichlet")
    # Température imposée : on définit ici les valeurs du signal
        instants = C.timeValues
        instants_positifs = instants[np.where(instants>=0.0)]
        sig_num = 15.0 + 5*np.cos(instants_positifs*1e-4)
        C.definir_signal("D",sig_num)
        print(C)
    else :
        C.definir_CL("D","Neumann")
        C.definir_signal("D",np.zeros_like)
    # Étape 4 - Résolution automatique dès que le problème est complet
    # Étape 5 - Tracés
    if C.tout_est_defini() :
        import matplotlib.pyplot as plt
        plt.figure("Température et densités surfaciques de flux",\
                   figsize=(12,8))
        ax_temp,ax_phi = plt.subplot(2,1,1),plt.subplot(2,1,2)
        plt.subplots_adjust(left=0.07, right=0.98, bottom=0.07, \
                            top=0.93, hspace=0.3)
        ax_phi.set_xlabel(r"Instant $t$ $[s]$",size=14)
        colors = [(0.8,0,0),(0,0.5,0),(0,0,1),(0.8,0.8,0),(1,0,1),\
                  (0,0.7,0.7),(0,0.3,0.6),(0.6,0.3,0),(0.8,0.3,0.3)]
        nbcol = len(colors)
        nocol = 0
        if choix != 4 :
            X = np.linspace(C.X[0],C.X[-1],5)
        else :
            X = np.array([C.X[0], 0.5*(C.X[0]+C.X[1]), \
                          C.X[1]-1e-4, C.X[1]+1e-4, \
                          0.5*(C.X[1]+C.X[2]), C.X[2]-1e-4, \
                          C.X[2]+1e-4,0.5*(C.X[2]+C.X[3]), \
                          C.X[3]])
        flux = []
        for x,clr in zip(X,colors) :
            Vt,T,phi = C.T_phi(x)
            ax_temp.plot(Vt, T, "-", color=clr, linewidth=2,\
                     label=r"$r={:.1f} mm$".format(1e3*x))
            ax_phi.plot(Vt, phi, "-", color=clr, linewidth=2,\
                     label=r"$r={:.1f} mm$".format(1e3*x))
        ax_temp.set_ylabel(r"Température $T(t)$ [°C]",size=14)
        ax_temp.grid() ; ax_phi.grid()
        ax_temp.legend(loc="center right",fontsize=12)       
        ax_phi.set_ylabel(r"Dens. surf. de flux " + \
                          "$\phi(t)$ $[W\cdot m^{-2}]$",size=14)
        plt.show()
