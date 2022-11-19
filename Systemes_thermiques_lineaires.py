# Version 1.24 - 2020, October, 29
# Project : SysTherLin (Systèmes thermiques linéaires)
# Author : Eric Ducasse
# License : CC-BY-NC
# Institution : ENSAM / I2M
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import numpy as np
from numpy.linalg import solve
from numpy.fft import irfft
from scipy.special import erf
from Couches_conductrices import Multicouche
#==================== CLASSE CAVITÉ ====================================
class Cavite :
    """Cavité dont les parois sont des multicouches 1D."""
    def __init__(self,volume,rho,Cp,parois,Tinit=0.0) :
        """ 'parois' est une liste de triplets (MC,côté,surface).
            'MC' est une instance de la classe Multicouche.
            'côté' vaut 'G' ou 'D' pour indiquer le côté de
            raccordement. 'surface' est la surface de contact entre
            la paroi et la cavité.
            'Tinit' désigne l'écart initial de température de la cavité
            par rapport à la température de référence.
            """
        self.__r = rho
        self.__c = Cp
        self.__v = volume
        self.__a = 1.0/(volume*rho*Cp)
        self.__Ti = Tinit
        self.__parois = []
        self.__S = []
        self.__n = []
        try :
            msg = ""
            for mc,GD,S in parois :
                if GD.lower() in ["g","gauche"] :
                    CLG = mc.CLG
                    if CLG[0] != "Convection" :
                        msg = "Constructeur de Cavite :: erreur : le "+\
                              "côté gauche du multicouche :\n"+\
                              mc.__str__()+\
                              "\n n'a pas une C.L. de convection."
                        raise
                    self.__n.append(0)
                elif GD.lower() in ["d","droite"] :
                    CLD = mc.CLD
                    if CLD[0] != "Convection" :
                        msg = "Constructeur de Cavite :: erreur : le "+\
                              "côté droit du multicouche :\n"+\
                              mc.__str__()+\
                              "\n n'a pas une C.L. de convection."
                        raise
                    self.__n.append(-1)
                else :
                    msg = "Constructeur de Cavite :: erreur : le côté "+\
                          "du multicouche n'est pas bien spécifié :"+\
                          "\n\t'G' ou 'D' et non pas '{}'".format(GD)
                    raise
                self.__parois.append(mc)
                self.__S.append(S)
        except :
            if msg == "" : msg = "Constructeur de Cavite :: erreur : "+\
                                 "'parois' n'est pas une liste de "+\
                                 "triplets (MC,côté,surface)"
            raise ValueError(msg)
    #++++++++++++++++++++
    @property
    def rho(self) : return self.__r
    @property
    def Cp(self) : return self.__c
    @property
    def volume(self) : return self.__v
    @property
    def a(self) :
        """T'(t) = a*flux total entrant."""
        return self.__a 
    @property
    def Tinit(self) : return self.__Ti
    @property
    def parois(self) : return tuple(self.__parois)
    @property
    def surfaces(self) : return tuple(self.__S)
    @property
    def cotes(self) :
        """0 pour gauche et -1 pour droite."""
        return tuple(self.__n)
    #++++++++++++++++++++
    def __str__(self) :
        nb_par = len(self.parois)
        if nb_par == 1 :
            msg = "Cavité à une seule paroi :"
        else :
            msg = f"Cavité à {nb_par} parois :"
        for nom,val,u in [["Volume",1e6*self.__v,"cm³"],
                    ["Masse volumique",self.rho,"kg/m^3"],
                    ["Capacité calorique",self.Cp,"J/K/kg"]] :
            msg += "\n\t{} : {:.2f} {}".format(nom,val,u)
        msg += "\n\tTempérature initiale : " + \
               "{:.2f} °C".format(self.Tinit)
        for no,S in enumerate(self.surfaces,1) :
            msg += f"\n\tSurface de la paroi n°{no} : " + \
                   f"{1e4*S:.2f} cm²"
        return msg         
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if __name__ == "__main__" :
    from Couches_conductrices import CoucheConductrice
    ### Exemple du TD1 de méthodes numériques 
    inox = CoucheConductrice(16.5,8000.0,500.0,3.0e-3,20.0)
    verre_bas =  CoucheConductrice(1.0,2800.0,1000.0,1.0e-3,20.0)
    socle = Multicouche([inox,verre_bas])
    socle.definir_CL("G","Neumann")
    socle.definir_signal("G",lambda s : 100/s) # W/m²
    socle.definir_CL("D","Convection",200.0)
    print("Socle :",socle)
    verre =  CoucheConductrice(1.0,2800.0,1000.0,8.0e-3,20.0)
    coque = Multicouche([verre])
    coque.definir_CL("G","Convection",200.0)
    coque.definir_CL("D","Convection",10.0)
    coque.definir_signal("D",lambda s : 20/s) # Température extérieure
    print("Coque :",coque)
    cavite = Cavite(0.2, 1000.0, 4200.0, \
                    [(socle,"D",0.4),(coque,"G",1.8)], 20.0)
    print(cavite)
#================ PETITE CLASSE UTILE ==================================
class orbites :
    def __init__(self,size) :
        self.__nb = size
        self.__orb = [[i] for i in range(size)]
    def add(self,pair) :
        n0,n1 = pair
        if not (0 <= n0 < self.__nb and 0 <= n1 < self.__nb) :
            return
        for i,e in enumerate(self.__orb) :
            if n0 in e : i0 = i
            if n1 in e : i1 = i
        if n0 == n1 : return # No new connection
        self.__orb[i0].extend(self.__orb[i1])
        self.__orb[i0].sort()
        self.__orb.pop(i1)
    @property
    def orbites(self) :
        return tuple([tuple(e) for e in self.__orb])        
    @property
    def nb(self) : return len(self.__orb)
    @property
    def all_connected(self) : return len(self.__orb) == 1
#================ CLASSE SYSTÈME THERMIQUE LINÉAIRE ====================
class SystemeThermiqueLineaire :
    """Système composé uniquement de multicouches 1D conductrices et
       de cavités."""
    def __init__(self,duree,dt,elements,calculer=False) :
        """elements est soit un seul multicouche, soit une seule cavité,
           soit une liste de plusieurs cavités dont les parois sont des
           multicouches."""
        if isinstance(elements,Multicouche) : # 1 seul multicouche
            self.__type = "un seul multicouche"
            MC = elements
            self.__LMC = [MC]
            MC.definir_temps(duree,dt)
            self.__LCAV = []
            self.__T_cav = []
            self.__time = MC.timeValues
            self.__s = MC.s
            if calculer : self.calculer_maintenant()
            self.__bornes,self.__dim = None,None
        elif isinstance(elements,Cavite) : # 1 cavité avec des parois
            self.__type = "une seule cavité"
            CAV = elements
            self.__LCAV = [CAV]
            self.__LMC = CAV.parois
            self.__bornes,pos = [0],0
            for mc in self.__LMC :
                mc.definir_temps(duree,dt)
                pos += 2*mc.nb
                self.__bornes.append(pos)
            self.__dim = pos + 1 # Taille de la matrice
            self.__time = mc.timeValues # Commun à tous
            self.__s = mc.s # idem
            if calculer : self.calculer_maintenant()
        else : # Plusieurs cavités
            self.__type = "plusieurs cavités"
            self.__LCAV = elements
            self.__LMC,no_mc = [],0 # Liste des multicouches
            # On attache « à la volée » à chaque multicouche des
            # attributs publics supplémentaires : 
            #      numero = son rang dans la liste globale
            #      no_cav = couple de booléens avec False si connexion
            #               avec une cavité et True sinon
            #               (numéro de cavité de façon temporaire)
            orb = orbites(len(self.__LCAV))
            for p,cav in enumerate(self.__LCAV) :
                for mc,gd in zip(cav.parois,cav.cotes) :
                    if mc not in self.__LMC :
                        mc.numero = no_mc # on numérote les multicouches
                        if gd == 0 : mc.no_cav = [p,-1]
                        else : mc.no_cav = [-1,p]
                        no_mc += 1
                        self.__LMC.append(mc)
                    else :
                        if mc.no_cav[gd] == -1 :
                            mc.no_cav[gd] = p
                            orb.add(mc.no_cav)
                        else :
                            msg = "Constructeur de SystemeThermiqueLi"+\
                                  "neaire :: Erreur : deux cavités du"+\
                                  " même côté du multicouche :\n"
                            msg += mc.__str__()
                            raise ValueError(msg)
            if not orb.all_connected :
                 msg = "Constructeur de SystemeThermiqueLineaire ::"+\
                       " Attention : {} sous-systèmes".format(orb.nb)+\
                       " découplés"
            for mc in self.__LMC : # Conversion en booléens
                for i in [0,1] : mc.no_cav[i] = (mc.no_cav[i] == -1)
            self.__bornes,pos = [0],0
            for mc in self.__LMC :
                mc.definir_temps(duree,dt)
                pos += 2*mc.nb
                self.__bornes.append(pos)
            self.__dim = pos + len(self.__LCAV) # Taille de la matrice
            self.__time = mc.timeValues # Commun à tous
            self.__s = mc.s # idem
            self.__T_cav = None # Liste des signaux de température dans
                                # les cavités, à calculer
            if calculer : self.calculer_maintenant()

    def calculer_maintenant(self, verbose=False):
        """Calcule la solution, si cela est possible."""
        print("SystemeThermiqueLineaire.calculer_maintenant...")
        if self.__type == "un seul multicouche" :
            MC = self.__LMC[0]
            if verbose : print("+++ Multicouche 1 :")
            if MC.tout_est_defini() :
                pass # tout est déjà calculé dans le multicouche
        elif self.__type == "une seule cavité" :
            s,bornes = self.__s,self.__bornes
            ns,dm = len(s),self.__dim
            big_M = np.zeros( (ns,dm,dm), dtype=complex)
            big_V = np.zeros( (ns,dm), dtype=complex)
            CAV = self.__LCAV[0]
            Tinit = CAV.Tinit
            OK = True
            for i,(mc,d,f,gd,S) in enumerate(zip(CAV.parois,\
                                             bornes[:-1],bornes[1:],\
                                             CAV.cotes,CAV.surfaces),\
                                             1) :
                if verbose : print("+++ Multicouche {} :".format(i))
                sgn = [True,True];sgn[gd]=False
                if not mc.tout_est_defini(signals=sgn) :
                    OK = False
                    continue
                big_M[:,d:f,d:f] = mc.M_matrix()
                layer = mc.couches[gd]
                if gd == 0 : # côté gauche
                    LP = layer.P(s,mc.X[0])[:,1,:] # DS Flux
                else : # côté droit
                    LP = layer.P(s)[:,1,:] # DS Flux
                P1 = np.einsum("i,ij->ij",CAV.a*S/s,LP)
                if gd == 0 : # côté gauche
                    big_M[:,d,-1] = -1.0 # Température
                    big_V[:,d] = (Tinit-mc.Tinit_gauche)/s
                                 # Saut initial de Température
                    big_M[:,-1,d:d+2] = -P1  # DS Flux
                    # autre côté :
                    big_V[:,f-1] = mc.TLsig("D",offset=False) 
                else : # côté droit
                    big_M[:,f-1,-1] = -1.0 # Température
                    big_V[:,f-1] = (Tinit-mc.Tinit_droite)/s
                                   # Saut initial de Température
                    big_M[:,-1,f-2:f] = P1  # DS Flux
                    # autre côté :
                    if mc.CLG[0] == '(Symétrie)' :
                        big_V[:,d] = 0.0
                    else :
                        big_V[:,d] = mc.TLsig("G",offset=False)
                #!!!
                # Bogue rectifié à partir de la version 1.23 :
                #       sauts internes de températures initiales
                TinitL = mc.couches[0].Tinit
                for nlay,layer in enumerate(mc.couches[1:],1) :
                    TinitR = layer.Tinit
                    big_V[:,d+2*nlay-1] += (TinitR-TinitL)/self.__s
                    TinitL = TinitR
                #!!!
                if verbose : print("\tOK")
            big_M[:,-1,-1] = -1.0
            if OK :
                V_AB = solve(big_M,big_V)
                for mc,d,f in zip(CAV.parois,bornes[:-1],bornes[1:]) :
                    mc.set_AB(V_AB[:,d:f])            
                self.__T_cav = [CAV.Tinit + mc.TLrec(V_AB[:,-1])]
                print("... Calcul effectué.")
        elif self.__type == "plusieurs cavités" :
            s,bornes = self.__s,self.__bornes
            ns,dm = len(s),self.__dim
            big_M = np.zeros( (ns,dm,dm), dtype=complex)
            big_V = np.zeros( (ns,dm), dtype=complex)
            OK = True
############
            for p,cav in enumerate(self.__LCAV,1) :
                if verbose :
                    print("+++ Cavité {} :".format(p))
                Tinit = cav.Tinit
                for mc,gd,S in zip(cav.parois,cav.cotes,cav.surfaces) :
                    no = mc.numero
                    np1 = no + 1
                    d,f = bornes[no:no+2]
                    if verbose :
                        print("+++ Multicouche {} :".format(np1))
                    if not mc.tout_est_defini(signals=mc.no_cav) :
                        OK = False
                        continue
                    big_M[:,d:f,d:f] = mc.M_matrix()
                    layer = mc.couches[gd]
                    if gd == 0 : # côté gauche
                        LP = layer.P(s,mc.X[0])[:,1,:]  # DS Flux
                    else : # côté droit
                        LP = layer.P(s)[:,1,:]  # DS Flux
                    P1 = np.einsum("i,ij->ij",cav.a*S/s,LP)
                    if gd == 0 : # côté gauche
                        big_M[:,d,-p] = -1.0
                        big_V[:,d] = (Tinit-mc.Tinit_gauche)/s
                                     # Saut initial de Température
                        big_M[:,-p,d:d+2] = -P1
                        # autre côté :
                        if mc.no_cav[-1] :
                            big_V[:,f-1] = mc.TLsig("D",offset=False)
                    else : # côté droit
                        big_M[:,f-1,-p] = -1.0
                        big_V[:,f-1] = (Tinit-mc.Tinit_droite)/s
                                       # Saut initial de Température
                        big_M[:,-p,f-2:f] = P1
                        # autre côté :
                        if mc.no_cav[0] :                            
                            if mc.CLG[0] == '(Symétrie)' :
                                big_V[:,d] = 0.0
                            else :
                                big_V[:,d] = mc.TLsig("G",offset=False)
                    if verbose : print("\tOK")
                big_M[:,-p,-p] = -1.0
############
            #!!!
            # Bogue rectifié à partir de la version 1.23 :
            #       sauts internes de températures initiales
            for i,mc in enumerate(self.__LMC,1) :
                # Différents multicouches
                print("Sauts de températures dans le multicouche " + \
                      "{}".format(i))
                no = mc.numero
                d = bornes[no]
                TinitL = mc.couches[0].Tinit
                for nlay,layer in enumerate(mc.couches[1:],1) :
                    TinitR = layer.Tinit
                    big_V[:,d+2*nlay-1] += (TinitR-TinitL)/s
                    TinitL = TinitR
            #!!!
            if OK :
                self.big_M = big_M
                self.big_V = big_V
                V_AB = solve(big_M,big_V) # Résolution du système
                for mc in self.__LMC : # Différents multicouches
                    no = mc.numero
                    d,f = bornes[no:no+2]
                    mc.set_AB(V_AB[:,d:f])
                # Températures dans les cavités
                self.__T_cav = [cav.Tinit + mc.TLrec(V_AB[:,-d]) \
                                for d,cav in enumerate(self.__LCAV,1)]
                print("... Calcul effectué.")
        else :
            print("SystemeThermiqueLineaire.calculer_maintenant ::"+\
                  "Type de système '{}' inconnu!".format(self.__type))
        return           
            
    @property
    def timeValues(self) : return self.__time
    @property
    def multicouches(self) : return self.__LMC
    @property
    def cavites(self) : return self.__LCAV
    @property
    def T_cavites(self) : return self.__T_cav
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if __name__ == "__main__" :
    import matplotlib.pyplot as plt
    choix = 5
    if choix == 1 : # Problème mal posé
        essai1 = SystemeThermiqueLineaire(1800.0, 1.0, socle, \
                                          calculer=True)
    elif choix == 2 : # Un multi-couche seul
        socle.definir_signal("D",lambda s: 15.0/s)
        essai2 = SystemeThermiqueLineaire(1800.0, 0.5, socle, \
                                          calculer=True)
        mc = essai2.multicouches[0]
        Vt,T,phi = mc.T_phi(mc.X[-1])
        ax_temp,ax_flux = plt.subplot(2,1,1),plt.subplot(2,1,2)
        plt.subplots_adjust(left=0.12,bottom=0.12,right=0.98,top=0.99)
        ax_temp.plot(essai2.timeValues,T,"-m",label="$T(t)$")
        ax_temp.grid() ; ax_temp.legend()
        ax_temp.set_ylabel("Température $T(t)$ [°C]")
        ax_flux.plot(essai2.timeValues,phi,"-b",label="$\phi(t)$")
        ax_flux.grid() ; ax_flux.legend()
        ax_flux.set_xlabel("Instants $t$ [s]")
        ax_flux.set_ylabel("$\phi(t)$ [W/m²]")
        plt.show()
    elif choix == 3 : # Exemple du TD n°1 de méthodes numériques
        essai3 = SystemeThermiqueLineaire(80*3600, 10.0, cavite, \
                                          calculer=True)
        plt.plot(essai3.timeValues/3600,essai3.T_cavites[0],"-m",\
                 label=r"$T_{\mathrm{eau}}$")
        socl = essai3.cavites[0].parois[0]
        _,T1,_ = socl.T_phi(0.5*socl.X[-1])
        plt.plot(essai3.timeValues/3600,T1,"-r",\
                 label=r"$T_{\mathrm{socle}}$")
        coqu = essai3.cavites[0].parois[1]
        _,T3,_ = coqu.T_phi(0.5*coqu.X[-1])
        plt.plot(essai3.timeValues/3600,T3,"-b",\
                 label=r"$T_{\mathrm{coque}}$")
        plt.legend(loc="best",fontsize=10)
        plt.grid() ; plt.show()
    elif choix == 4 : # Cylindre d'acier
        from Couches_conductrices import CoucheConductriceCylindrique
        acier=CoucheConductriceCylindrique(50.2,7.85e3,1000,0,0.05,50.0)
                                                      # T° initiale 50°C
        cylindre = Multicouche([acier])
        cylindre.definir_CL("D","Convection",200.0) # Plongé dans l'eau
        cylindre.definir_signal("D",lambda s: 10/s) # T° ext. 10°C
        essai4 = SystemeThermiqueLineaire(2*3600, 10.0, cylindre, \
                                          calculer=True)
        Vt,T_ext,F_ext = cylindre.T_phi(0.05)
        _,T_mil,F_mil = cylindre.T_phi(0)
        ax_temp,ax_flux = plt.subplot(2,1,1),plt.subplot(2,1,2)
        plt.subplots_adjust(left=0.12,bottom=0.12,right=0.98,top=0.99)
        ax_temp.plot(Vt/60,T_ext,"-b",label=r"$T_{\mathrm{ext}}$")
        ax_temp.plot(Vt/60,T_mil,"-m",label=r"$T_{\mathrm{mil}}$")
        ax_temp.legend(loc="best",fontsize=10)
        ax_temp.set_ylabel("Température $T(t)$ [°C]")
        ax_flux.plot(Vt/60,F_ext,"--r",label=r"$\phi_{\mathrm{ext}}$")
        ax_flux.legend(loc="best",fontsize=10)
        ax_flux.set_xlabel("Instants $t$ [s]")
        ax_flux.set_ylabel("$\phi(t)$ [W/m²]")
        ax_temp.grid() ; ax_flux.grid()
        plt.show()
    elif choix == 5 : # Cylindre d'acier à 50°C plongé dans une cavité
                      #  d'eau à 10°C, avec un air extérieur à 20°C
        from Couches_conductrices import CoucheConductriceCylindrique
        acier=CoucheConductriceCylindrique(50.2, 7.85e3, 1000, 0, \
                                           0.05, 50.0)
        cylindre = Multicouche([acier])
        cylindre.definir_CL("D","Convection",200.0) # cavité d'eau
        acier2 = CoucheConductriceCylindrique(50.2, 7.85e3, 1000, 0.1, \
                                              0.12, 10.0)
        coque = Multicouche([acier2])
        coque.definir_CL("G","Convection",200.0) # cavité d'eau
        coque.definir_CL("D","Convection",20.0)  # air extérieur
        coque.definir_signal("D",lambda s: 20.0/s)
        R_cav_min,R_cav_max = cylindre.X[-1],coque.X[0]
        V_sur_L = np.pi*(R_cav_max**2+R_cav_min**2)
        Smin_sur_L, Smax_sur_L = 2*np.pi*R_cav_min,2*np.pi*R_cav_max
        cavite = Cavite(V_sur_L, 1000.0, 4200.0, \
                        [(cylindre,"D",Smin_sur_L), \
                         (coque,"G",Smax_sur_L)], 10.0)
        essai5 = SystemeThermiqueLineaire(6.5*3600, 5.0, cavite, \
                                          calculer=True)
        Vt,T_ext,F_ext = cylindre.T_phi(0.05)
        _,T_mil,F_mil = cylindre.T_phi(0)
        _,T_coq,F_coq = coque.T_phi(coque.X[-1])
        T_eau = essai5.T_cavites[0]
        ax_temp,ax_flux = plt.subplot(2,1,1),plt.subplot(2,1,2)
        plt.subplots_adjust(left=0.12,bottom=0.12,right=0.98,top=0.99)
        ax_flux.set_xlabel(r"Instant $t\;[h]$")
        ax_temp.plot(Vt/3600, T_eau, "-b", \
                     label=r"$T_{\mathrm{eau}}(t)$")
        ax_temp.plot(Vt/3600, T_ext, "-g", \
                     label=r"$T_{\mathrm{ext}}(t)$")
        ax_temp.plot(Vt/3600, T_mil, "-m", \
                     label=r"$T_{\mathrm{mil}}(t)$")
        ax_temp.plot(Vt/3600, T_coq, "-r", \
                     label=r"$T_{\mathrm{coq}}$")
        ax_temp.set_ylabel(r"Température [°C]")
        ax_temp.legend(loc="best",fontsize=10) ; ax_temp.grid()
        ax_flux.plot(Vt/3600, F_ext, "--b", \
                     label=r"$\phi_{\mathrm{ext}}(t)$")
        ax_flux.plot(Vt/3600, F_coq, "--r" ,\
                     label=r"$\phi_{\mathrm{coq}}(t)$")
        ax_flux.set_ylabel(r"D.S. Flux [W/m²]")
        ax_flux.legend(loc="best",fontsize=10) ; ax_flux.grid()
        plt.show()

