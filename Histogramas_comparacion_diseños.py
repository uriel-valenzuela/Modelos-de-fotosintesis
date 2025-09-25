"Hace los histogramas comparando los tres disenos"

#%%
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from pytwalk import BUQ
from FvCB_V11 import PHSmodel
from DataDict import init_dict
from readdata import *
from scipy.stats import bernoulli, weibull_min, norm, gamma, expon, uniform
from numpy import loadtxt, linspace, outer, ones
from matplotlib.pylab import subplots, plot, close, hist, show,rcParams, axvline, title, scatter, fill_between, tight_layout

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
for N in range(1, 37):  #N es el numero de planta
    if N!=17: #No hay valores de referencia para la planta 17
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
        Data = np.array([dat['I'], dat['PCi'], dat['O'], dat['T'], dat['Osp']]) 
        sigma = np.array(dat['Error'])
        
        dat = modelo.dataAll()
        
        #%%
        "Datos para el diseno de zigzag"
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

        #Diseño convencional
        buq = BUQ( q = q, data = None, logdensity=logdensity,simdata=simdata, sigma=sigma,\
             F = F, t = Data, par_names = par_names, par_prior = par_prior, par_supp=par_supp)
        
        #Diseño de rejilla    
        buqgrid = BUQ( q = q, data = None, logdensity=logdensity,simdata=simdata, sigma=sigma_grid,\
            F = F, t = Data_grid, par_names = par_names, par_prior = par_prior, par_supp=par_supp)
            
        #Diseño de zigzag
        buqzigzag = BUQ( q = q, data = None, logdensity=logdensity,simdata=simdata, sigma=sigma_zz,\
            F = F, t = Data_zz, par_names = par_names, par_prior = par_prior, par_supp=par_supp)
        
        #%%
        "Simulacion de datos sinteticos"
        #Valores de referencia para los parametros
        df = pd.read_excel('./Parametros_ref.xlsx', header=[0]) #Lee el excel con los valores de referencia
        x_ref = df[f"Planta {N}"]  #Toma el valor de referencia para la planta N (Yin et al p.41-44)
        
        #Valores sinteticos para ver los valores de referencia
        buq.SimData(x = x_ref) 
        buqzigzag.SimData(x = x_ref)
        buqgrid.SimData(x = x_ref)
        
        #Cargar el MCMC ya guardado
        buq.LoadtwalkOutput(f"MCMC_conv{N}.csv1")
        buqgrid.LoadtwalkOutput(f"MCMC_grid{N}.csv1") 
        buqzigzag.LoadtwalkOutput(f"MCMC_zigzag{N}.csv1") 
        
        #%%
        "Histogramas de los parámetros"
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
        fig, axes = subplots(nrows=5, ncols=3, figsize=(12, 15)) #Cuadricula de 5x3 para hacer los histogramas juntos
        axes = axes.flatten()  #Para poder indexarlos linealmente

        for k in range(15): #Son 15 parametros
            ax = axes[k] #Llama a la k-esima entrada de la cuadricula
            #Histogramas rellenos graficados en la entrada k de la cuadricula
            buq.PlotPost(par=k, bins=15,histtype='stepfilled', facecolor='gray', alpha=0.1, burn_in=1_000_000, ax=ax)
            buqgrid.PlotPost(par=k, bins=15,histtype='stepfilled', facecolor='gray', alpha=0.1, burn_in=1_000_000, ax=ax)
            buqzigzag.PlotPost(par=k, bins=15,histtype='stepfilled', facecolor='gray', alpha=0.1, burn_in=1_000_000, ax=ax)
            #Histogramas con contornos de colores graficados en la entrada k de la cuadricula
            buq.PlotPost(par=k, bins=15,histtype='step', edgecolor='red', linewidth=1.5, burn_in=1_000_000, ax=ax)
            buqgrid.PlotPost(par=k, bins=15,histtype='step', edgecolor='brown', linewidth=1.5, burn_in=1_000_000, ax=ax)
            buqzigzag.PlotPost(par=k, bins=15,histtype='step', edgecolor='blue', linewidth=1.5, burn_in=1_000_000, ax=ax)
            # Agregar letra del inciso
            letra = chr(97 + k)  # 97 es el código ASCII de 'a'
            ax.set_title(f"({letra})", loc='center') #Pone el inciso a cada histograma
        
        #Borrar los ejes sobrantes
        for j in range(k+1, len(axes)):
            fig.delaxes(axes[j]) 
            tight_layout()
            
        #Guarda la imagen en la carpeta y con el nombre para cada planta
        fig.savefig(f"Histogramas/Comparacion de disenos/{plant_names[N]}.png", dpi=300, bbox_inches='tight')
        
