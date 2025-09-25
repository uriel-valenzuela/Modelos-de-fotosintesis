"Hace los histogramas del experimento de discernir entre genotipos"

#%%
import numpy as np
import pandas as pd
from pytwalk import BUQ
from FvCB_V11 import PHSmodel
from DataDict import init_dict
from readdata import *
from scipy.stats import bernoulli, weibull_min, norm, gamma, expon, uniform
from numpy import loadtxt, linspace, outer, ones
from matplotlib.pylab import subplots, plot, close, hist, show, axvline, title, tight_layout, axvline, scatter, rcParams
from matplotlib.patches import Patch

#%%
"Configuracion para el diseno de zigzag"
Czz = np.linspace(1300,0,61) #CO2
Izz = np.full(61,2000) #luz incidente
for i in range(10):
    for j in [0,20,40]:
        Izz[i+j] = 2000 - i*200
for i in range(10,20):
    for j in [0,20,40]:
        Izz[i+j] = (i-10)*200

#Error para el diseno de zigzag
zigzag = pd.read_excel('./Zigzag.xlsx', header=[0]) #Lee el excel con los errores para el diseno de zigzag
sigma_zz = zigzag["Error"]

#%%     
"Configuracion para el diseno de rejilla"     
Igrid = outer( linspace( 0, 2000, num=20), ones(20)) #Luz incidente
Cgrid = outer( linspace( 0., 1300, num=20), ones(20)).T #Niveles de CO2
Igrid = np.array([Igrid.ravel(),Igrid.ravel()]).ravel()
Cgrid = np.array([Cgrid.ravel(),Cgrid.ravel()]).ravel()

#Error para el diseno de rejilla
grid = pd.read_excel('./Grid.xlsx', header=[0]) #Lee el excel con los errores para el diseno de rejilla
sigma_grid = grid["Error"]

#%%
q = 15 # Numero de parnmetros a inferir
logdensity=norm.logpdf #log-densidad del modelo
simdata = lambda n, loc, scale: norm.rvs( size=n, loc=loc, scale=scale) #Simula datos con la distribucion normal
#Nombres de los parametros
par_names=["gmo", "Vcmax", "delta", "Tp", r"$\theta_{NPR}$",
            r"$\theta_{PR}$", r"$J_{max}^{NPR}$", 
            r"$J_{max}^{PR}$", r"$k2_{LL}^{NPR}$" , 
            r"$k2_{LL}^{PR}$", "Sco", "Kmo", "Kmc", 
            r"$Rd_{NPR}$", r"$Rd_{PR}$"]
               
#VarsOrder =  gmo, Vcmax, delta, Tp, thetaNPR, thetaPR, JmaxNPR, JmaxPR, k2_LLNPR, k2_LLPR, Sco, Kmo, Kmc, Rd_NPR, Rd_PR 

#Nombres de las muestras
plant_names = ["P1L10A","P2L3D","P3L5B","P4L9B", "P5L11B","P6L2E","P7L8B","P8L1D","P9L12E","P10L4E","P11L7C","P12L6C",
               "P13L5D","P14L9A","P15L10C","P16L3E","P17L8A","P18L1B","P19L11C","P20L2B","P21L1A","P22L6D","P23L12D","P24L4A",
               "P25L9D","P26L10E","P27L3A","P28L5C","P29L1C","P30L11E","P31L2A","P32L8E","P33L6B","P34L12C","P35L4C","P36L7D"]

#A prioris de los parametros
par_prior =[uniform(0.0, scale=0.01),    #gmo,
            uniform(10.0,scale=190),      #Vcmax
            uniform(0.0, scale=5.0),      #delta
            uniform(0.0, scale=30),      #Tp 
            uniform(0.0, scale=1.0),     #ThetaNPR
            uniform(0.0, scale=1.0),     #ThetaPR
            uniform(2.0, scale=498),     #JmaxNPR
            uniform(2.0, scale=498),     #JmaxPR,
            uniform(0.1, scale=0.8),     #k2_LLNPR,
            uniform(0.1, scale=0.8),     #k2_LL_PR,
            uniform(0.0, scale=11.0),     #Sco
            uniform(1.0, scale=69),      #Kmc, limites en Pa 
            uniform(1.0, scale=69),      #Kmo, limites en kPa 
            uniform(0.0, scale=10),      #Rd_NPR
            uniform(0.0, scale=10)      #Rd_PR 
            ] 

#Soporte de las a priori 
par_supp  = [ lambda al: al>0.0, lambda la: la>0.0, lambda al: al>0.0, lambda la: la>0.0, lambda al: al>0.0,
              lambda al: al>0.0, lambda la: la>0.0, lambda al: al>0.0, lambda la: la>0.0, lambda al: al>0.0,
              lambda al: al>0.0, lambda la: la>0.0, lambda al: al>0.0, lambda la: la>0.0, lambda la: la>0.0]
               

#%%
"Bucle para realizar los histogramas para cada planta"
for N in range(1,37): #N es el numero de planta
    if N != 17: #No hay valores de referencia para la planta 17
        modelo = PHSmodel(init_dict)
        datos_trigo = datosendic(oxigeno = [20, 210], N = N)
        modelo.uploadExpData(datos_trigo)
        #Filtra los datos quitando los primeros 5 
        datos1=modelo.dataAt(constvar='Qin',oxigeno=20)
        datos2=modelo.dataAt(constvar='Qin',oxigeno=210)
        datos3=modelo.dataAt(constvar='Ci',oxigeno=20)
        datos4=modelo.dataAt(constvar='Ci',oxigeno=210)
        #Agrega identificadores para los niveles de oxigeno
        datos1['Osp'] = np.full(len(datos1["A"]),20)
        datos2['Osp'] = np.full(len(datos2["A"]),210)
        datos3['Osp'] = np.full(len(datos3["A"]),20)
        datos4['Osp'] = np.full(len(datos4["A"]),210)
        
        #Crea diccionario concantenando las variables en cada una de las mediciones
        dat = {
        key: np.concatenate((datos1[key], datos2[key], datos3[key], datos4[key]))
        for key in datos1}

        # Set-up experimental data
        Data = np.array([dat['I'], dat['C'], dat['O'], dat['T'], dat['Osp']]) 
        sigma = np.array(dat['Error'])

        #%%
        "Datos para el diseno de zigzag"
        dat = modelo.dataAll()

        # Set-up experimental data
        #Temperatura de la hoja
        T_NPR = np.full(61,np.mean(datos_trigo["e1"]["Tleaf"]))
        T_PR = np.full(61,np.mean(datos_trigo["e2"]["Tleaf"]))
        #oxigeno
        O_NPR = np.full(61,np.mean(datos_trigo["e1"]["PO2"]))
        O_PR = np.full(61,np.mean(datos_trigo["e2"]["PO2"]))
        #Niveles de oxigeno
        Osp_NPR = np.full(61,20)
        OSP_PR = np.full(61,210)
        
        # Set-up experimental data (zigzag)
        Data_zz = np.array([np.array([Izz,Izz]).ravel(),
                            np.array([Czz,Czz]).ravel(),
                            np.array([T_NPR,T_PR]).ravel(),
                            np.array([O_NPR,O_PR]).ravel(),
                            np.array([Osp_NPR,OSP_PR]).ravel()])

        #%%
        "Datos para el diseno de rejilla"
        # Set-up experimental data
        mask = dat['Osp'] == 20 #Filtrar los datos de oxigeno bajo
        #Niveles de oxigeno
        O = np.array([np.full(int(len(Cgrid)/2),np.mean(dat['O'][mask])),np.full(int(len(Cgrid)/2),np.mean(dat['O'][~mask]))]).ravel()
        #Temperatura
        T = np.array(np.full(len(Cgrid),np.mean(dat['T']))).ravel()
        #Identificador de nivel de oxigeno
        Osp = np.array([np.full(int(len(Cgrid)/2), 20),np.full(int(len(Cgrid)/2),210)]).ravel()
        
        # Set-up experimental data (rejilla)
        Data_grid = np.array([Igrid, Cgrid, O, T, Osp]) 
        
        #%%
        ### Define the Forward map with signature F( theta, Data)
        def F(theta, Data):
            gmo, Vcmax, delta, Tp, thetaNPR, thetaPR, JmaxNPR, JmaxPR, k2_LLNPR, k2_LLPR, Sco, Kmo, Kmc, Rd_NPR, Rd_PR = theta
            P = Data
            A = modelo.fABayes(P, gmo, Vcmax, delta, Tp, thetaNPR, thetaPR, JmaxNPR, JmaxPR, k2_LLNPR, k2_LLPR, Sco, Kmo, Kmc, Rd_NPR, Rd_PR)
            return A
        #%%
        "Simulacion de datos sinteticos"
        #Valores de referencia para los parametros
        df = pd.read_excel('./Parametros_ref.xlsx', header=[0]) #Lee el excel con los valores de referencia
        x_ref = df[f"Planta {N}"]  #Toma el valor de referencia para la planta N (Yin et al p.41-44)
        
        #Cambiar los valores de V_cmax
        V_30 = x_ref.copy()
        V_30[1] = 30
        V_50 = x_ref.copy()
        V_50[1] = 50
        V_100 = x_ref.copy()
        V_100[1] = 100
        V_150 = x_ref.copy()
        V_150[1] = 150
        V_35 = x_ref.copy()
        V_35[1] = 35
        V_55 = x_ref.copy()
        V_55[1] = 55 
        
        #Listas para guardar los valores de los parametros de referencia
        dVcmax = [V_30,V_50,V_100,V_150] 
        dVcmax2 = [V_35,V_55]
        #%%
        "Lectura de los MCMC (distancias largas)"
        #Listas para guardar los MCMC
        lbuq = []
        lbuqzz = []
        lbuq_sim = []
        
        #Valores con distancias largas
        for V in dVcmax:
            #Valor de V_cmax
            i = V[1]
            "Diseno de zigzag"
            buqzz_i = BUQ( q = q, data = None, logdensity=logdensity,simdata=simdata, sigma=sigma_zz,\
                F = F, t = Data_zz, par_names = par_names, par_prior = par_prior, par_supp=par_supp)
            buqzz_i.SimData(x = V)
            buqzz_i.LoadtwalkOutput(f"MCMCzigzag_V{i}_{N}.csv1") 
            "Diseno convencional"
            buq_i = BUQ( q = q, data = None, logdensity=logdensity,simdata=simdata, sigma=sigma,\
                F = F, t = Data, par_names = par_names, par_prior = par_prior, par_supp=par_supp)
            buq_i.SimData(x = V)
            buq_i.LoadtwalkOutput(f"MCMCconv_V{i}_{N}.csv1") 
            "Diseno de rejilla"
            buq_sim_i = BUQ( q = q, data = None, logdensity=logdensity,simdata=simdata, sigma=sigma_grid,\
                F = F, t = Data_grid, par_names = par_names, par_prior = par_prior, par_supp=par_supp)
            buq_sim_i.SimData(x = V)
            buq_sim_i.LoadtwalkOutput(f"MCMCgrid_V{i}_{N}.csv1") 
            #Agregar MCMC a las listas
            lbuqzz.append(buqzz_i)
            lbuq.append(buq_i)
            lbuq_sim.append(buq_sim_i)
        
        
        #%%
        "Lectura de los MCMC (distancias cortas)"
        #Listas para guardar los MCMC
        lbuq2 = []
        lbuqzz2 = []
        lbuq_sim2 = []
        
        #Valores con distancias cortas
        for V in dVcmax2:
            #Valor de V_cmax
            i = V[1]
            "Diseno de zigzag"
            buqzz_i = BUQ( q = q, data = None, logdensity=logdensity,simdata=simdata, sigma=sigma_zz,\
                F = F, t = Data_zz, par_names = par_names, par_prior = par_prior, par_supp=par_supp)
            buqzz_i.SimData(x = V)
            buqzz_i.LoadtwalkOutput(f"MCMCzigzag_V{i}_{N}.csv1")
            "Diseno convencional"
            buq_i = BUQ( q = q, data = None, logdensity=logdensity,simdata=simdata, sigma=sigma,\
                F = F, t = Data, par_names = par_names, par_prior = par_prior, par_supp=par_supp)
            buq_i.SimData(x = V)
            buq_i.LoadtwalkOutput(f"MCMCconv_V{i}_{N}.csv1")
            "Diseno de rejilla"
            buq_sim_i = BUQ( q = q, data = None, logdensity=logdensity,simdata=simdata, sigma=sigma_grid,\
                F = F, t = Data_grid, par_names = par_names, par_prior = par_prior, par_supp=par_supp)
            buq_sim_i.SimData(x = V)
            buq_sim_i.LoadtwalkOutput(f"MCMCgrid_V{i}_{N}.csv1") 
            #Agregar MCMC a las listas
            lbuqzz2.append(buqzz_i)
            lbuq2.append(buq_i)
            lbuq_sim2.append(buq_sim_i)
        
        #%%
        "MCMC de las distancias"
        #Listas con las distancias
        distancias = [20,70,120,50,100,50] #Largas
        distancias2 = [5,5] #Cortas
        
        #Listas para guardar los MCMC de las distancias (largas)
        "Convencional"
        d50_30p = []
        d100_30p = []
        d150_30p = []

        d100_50p = []
        d150_50p = []

        d150_100p = []
        
        "Zigzag"
        dzz50_30p = []
        dzz100_30p = []
        dzz150_30p = []

        dzz100_50p = []
        dzz150_50p = []

        dzz150_100p = []
        
        "Rejilla"
        dsim50_30p = []
        dsim100_30p = []
        dsim150_30p = []

        dsim100_50p = []
        dsim150_50p = []

        dsim150_100p = []
        
        #Listas para guardar los MCMC de las distancias (cortas)
        "Convencional"
        d35_30p = []
        d55_50p = []
        
        "Zigzag"
        dzz35_30p = []
        dzz55_50p = []
        
        "Rejilla"
        dsim35_30p = []
        dsim55_50p = []
        
        #MCMC de las distancias
        for k in range(1_000_000,len(lbuq[0].Output[:,0])):
            d50_30p.append(np.linalg.norm(lbuq[0].Output[k,1]-lbuq[1].Output[k,1]))
            d100_30p.append(np.linalg.norm(lbuq[0].Output[k,1]-lbuq[2].Output[k,1]))
            d150_30p.append(np.linalg.norm(lbuq[0].Output[k,1]-lbuq[3].Output[k,1]))
            
            d100_50p.append(np.linalg.norm(lbuq[1].Output[k,1]-lbuq[2].Output[k,1]))
            d150_50p.append(np.linalg.norm(lbuq[1].Output[k,1]-lbuq[3].Output[k,1]))
            
            d150_100p.append(np.linalg.norm(lbuq[2].Output[k,1]-lbuq[3].Output[k,1]))
            
            dzz50_30p.append(np.linalg.norm(lbuqzz[0].Output[k,1]-lbuqzz[1].Output[k,1]))
            dzz100_30p.append(np.linalg.norm(lbuqzz[0].Output[k,1]-lbuqzz[2].Output[k,1]))
            dzz150_30p.append(np.linalg.norm(lbuqzz[0].Output[k,1]-lbuqzz[3].Output[k,1]))
            
            dzz100_50p.append(np.linalg.norm(lbuqzz[1].Output[k,1]-lbuqzz[2].Output[k,1]))
            dzz150_50p.append(np.linalg.norm(lbuqzz[1].Output[k,1]-lbuqzz[3].Output[k,1]))
            
            dzz150_100p.append(np.linalg.norm(lbuqzz[2].Output[k,1]-lbuqzz[3].Output[k,1]))
            
            dsim50_30p.append(np.linalg.norm(lbuq_sim[0].Output[k,1]-lbuq_sim[1].Output[k,1]))
            dsim100_30p.append(np.linalg.norm(lbuq_sim[0].Output[k,1]-lbuq_sim[2].Output[k,1]))
            dsim150_30p.append(np.linalg.norm(lbuq_sim[0].Output[k,1]-lbuq_sim[3].Output[k,1]))
            
            dsim100_50p.append(np.linalg.norm(lbuq_sim[1].Output[k,1]-lbuq_sim[2].Output[k,1]))
            dsim150_50p.append(np.linalg.norm(lbuq_sim[1].Output[k,1]-lbuq_sim[3].Output[k,1]))
            
            dsim150_100p.append(np.linalg.norm(lbuq_sim[2].Output[k,1]-lbuq_sim[3].Output[k,1]))
            
            d35_30p.append(np.linalg.norm(lbuq[0].Output[k,1]-lbuq2[0].Output[k,1]))
            d55_50p.append(np.linalg.norm(lbuq[1].Output[k,1]-lbuq2[1].Output[k,1]))
            
            dzz35_30p.append(np.linalg.norm(lbuqzz[0].Output[k,1]-lbuqzz2[0].Output[k,1]))
            dzz55_50p.append(np.linalg.norm(lbuqzz[1].Output[k,1]-lbuqzz2[1].Output[k,1]))
            
            dsim35_30p.append(np.linalg.norm(lbuq_sim[0].Output[k,1]-lbuq_sim2[0].Output[k,1]))
            dsim55_50p.append(np.linalg.norm(lbuq_sim[1].Output[k,1]-lbuq_sim2[1].Output[k,1]))


        #%%
        #Guardar los MCMC de las distancias
        dconv = [d50_30p,d100_30p,d150_30p,d100_50p,d150_50p,d150_100p] #Convencional
        dzz = [dzz50_30p,dzz100_30p,dzz150_30p,dzz100_50p,dzz150_50p,dzz150_100p] #Zigzag
        dsim = [dsim50_30p,dsim100_30p,dsim150_30p,dsim100_50p,dsim150_50p,dsim150_100p] #Rejilla
        dconv2 = [d35_30p,d55_50p] #Convencional
        dzz2 = [dzz35_30p,dzz55_50p] #Zigzag
        dsim2 = [dsim35_30p,dsim55_50p] #Rejilla

        # %%
        "Nombres de las distancias"
        d_names = ["50-30","100-30","150-30","100-50","150-50","150-100"]
        d_names2 = ["35-30","55-50"]
        
        # %%
        "Histogramas de V_cmax"
        #Estilo y tamanos de fuentes
        rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.size": 12,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12})
        fig, axes = subplots(nrows=2, ncols=2, figsize=(12, 15)) #Cuadricula de 2x2 para hacer los histogramas juntos
        axes = axes.flatten() #Para poder indexarlos linealmente
        
        
        for k in range(4):
            #Histogramas rellenos
            lbuq[k].PlotPost(par=1, bins=15, burn_in=1_000_000,histtype='stepfilled', facecolor='gray', alpha=0.1,ax=axes[k])
            lbuqzz[k].PlotPost(par=1, bins=15, burn_in=1_000_000,histtype='stepfilled', facecolor='gray', alpha=0.1,ax=axes[k])
            lbuq_sim[k].PlotPost(par=1, bins=15, burn_in=1_000_000,histtype='stepfilled', facecolor='gray', alpha=0.1,ax=axes[k])
            #Histogramas con contornos de colores
            lbuq[k].PlotPost(par=1, bins=15, burn_in=1_000_000,histtype='step', edgecolor='red', linewidth=1.5, ax=axes[k])
            lbuqzz[k].PlotPost(par=1, bins=15, burn_in=1_000_000,histtype='step', edgecolor='blue', linewidth=1.5, ax=axes[k])
            lbuq_sim[k].PlotPost(par=1, bins=15, burn_in=1_000_000,histtype='step', edgecolor='brown', linewidth=1.5, ax=axes[k])
            axes[k].set_title(r"$V_{{cmax}}=" + str(int(dVcmax[k][1])) + "$", fontsize=12) #Nombre con el valor real de V_{cmax}
        #Guardar la imagen con los histogramas
        fig.savefig(f"Histogramas/Discernir genotipos/Vcmax/{plant_names[N-1]}.png", dpi=300, bbox_inches='tight')    
        
        #%%
        """
        "Histogramas V_cmax (para distancias cortas)"
        rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.size": 12,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12})
        fig, axes = subplots(nrows=2, ncols=2, figsize=(12, 15))
        axes = axes.flatten()


        for k in range(4):
            lbuq[k].PlotPost(par=1, bins=15, burn_in=1_000_000,histtype='stepfilled', facecolor='gray',alpha=0.1,ax=axes[k])
            lbuqzz[k].PlotPost(par=1, bins=15, burn_in=1_000_000,histtype='stepfilled', facecolor='gray', alpha=0.1,ax=axes[k])
            lbuq_sim[k].PlotPost(par=1, bins=15, burn_in=1_000_000,histtype='stepfilled', facecolor='gray',alpha=0.1,ax=axes[k])
            lbuq[k].PlotPost(par=1, bins=15, burn_in=1_000_000,histtype='step', edgecolor='red', linewidth=1.5, ax=axes[k])
            lbuqzz[k].PlotPost(par=1, bins=15, burn_in=1_000_000,histtype='step', edgecolor='blue',linewidth=1.5, ax=axes[k])
            lbuq_sim[k].PlotPost(par=1, bins=15, burn_in=1_000_000,histtype='step', edgecolor='brown',linewidth=1.5, ax=axes[k])
            axes[k].set_title(r"$V_{{cmax}}=" + str(int(dVcmax[k][1])) + "$", fontsize=12)
        fig.savefig(f"Histogramas Vcmax cortas {N}.png", dpi=300, bbox_inches='tight')
        """
        
        #%%
        "Histogramas distancias largas"
        #Estilo y tamanos de fuentes
        rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.size": 12,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12})
        fig, axes = subplots(nrows=3, ncols=2, figsize=(12, 15)) #Cuadricula de 3x2 para hacer los histogramas juntos
        axes = axes.flatten() #Para poder indexarlos linealmente
        for k in range(6):
            #Histogramas de las distancias largas rellenos
            axes[k].hist(dconv[k][1_000_000:], bins=15, density=True, histtype='stepfilled', facecolor='gray', alpha=0.1)
            axes[k].hist(dconv[k][1_000_000:], bins=15, density=True, histtype='step', edgecolor='red', linewidth=1.5)
            axes[k].hist(dzz[k][1_000_000:], bins=15, density=True, histtype='stepfilled', facecolor='gray', alpha=0.1)
            #Histogramas de las distancias largas con contornos de colores
            axes[k].hist(dzz[k][1_000_000:], bins=15, density=True, histtype='step', edgecolor='blue', linewidth=1.5)
            axes[k].hist(dsim[k][1_000_000:], bins=15, density=True, histtype='stepfilled', facecolor='gray', alpha=0.1)
            axes[k].hist(dsim[k][1_000_000:], bins=15, density=True, histtype='step', edgecolor='brown', linewidth=1.5)
            axes[k].set_title(f"{d_names[k]}", fontsize=12) #Nombres de las distancias
            axes[k].axvline(x=distancias[k],color="black") #Valor real de la distancia
        #Guardar la imagen con los histogramas
        fig.savefig(f"Histogramas/Discernir genotipos/Comparaciones Distancias/{plant_names[N-1]}.png", dpi=300, bbox_inches='tight')
        
        #%%
        "Histogramas distancias cortas"
        #Estilo y tamanos de fuentes
        rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.size": 12,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12})
        fig, axes = subplots(nrows=1, ncols=2, figsize=(15, 7)) #Cuadricula de 1x2 para hacer los histogramas juntos
        axes = axes.flatten() #Para poder indexarlos linealmente
        for k in range(2):
            axes[k].hist(dconv2[k][1_000_000:], bins=15, density=True, histtype='stepfilled', facecolor='gray', alpha=0.1)
            axes[k].hist(dconv2[k][1_000_000:], bins=15, density=True, histtype='step', edgecolor='red', linewidth=1.5)
            axes[k].hist(dzz2[k][1_000_000:], bins=15, density=True, histtype='stepfilled', facecolor='gray', alpha=0.1)
            axes[k].hist(dzz2[k][1_000_000:], bins=15, density=True, histtype='step', edgecolor='blue', linewidth=1.5)
            axes[k].hist(dsim2[k][1_000_000:], bins=15, density=True, histtype='stepfilled', facecolor='gray', alpha=0.1)
            axes[k].hist(dsim2[k][1_000_000:], bins=15, density=True, histtype='step', edgecolor='brown', linewidth=1.5)
            axes[k].set_title(f"{d_names2[k]}", fontsize=12)
            axes[k].axvline(x=distancias2[k],color="black")
        #Guardar la imagen con los histogramas
        fig.savefig(f"Histogramas/Discernir genotipos/Comparaciones Distancias cortas/{plant_names[N-1]}.png", dpi=300, bbox_inches='tight')

        #%%




        

