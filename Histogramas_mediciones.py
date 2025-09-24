"""
Hacer los histogramas de los parámetros usando las mediciones
"""

#%%
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from pytwalk import BUQ
from FvCB_V11 import PHSmodel
from DataDict import init_dict
from readdata import *
from scipy.stats import bernoulli, weibull_min, norm, gamma, expon, uniform
from matplotlib.pylab import subplots, plot, close, hist, show,rcParams, axvline, title, scatter, fill_between, tight_layout

#%%
q = 15 # Numero de parnmetros a inferir
logdensity=norm.logpdf #log-densidad del modelo
#Nombres de los parÃ¡metros
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

#A prioris de los parámetros
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


#Soportes de las a priori
par_supp  = [ lambda al: al>0.0, lambda la: la>0.0, lambda al: al>0.0, lambda la: la>0.0, lambda al: al>0.0,
              lambda al: al>0.0, lambda la: la>0.0, lambda al: al>0.0, lambda la: la>0.0, lambda al: al>0.0,
              lambda al: al>0.0, lambda la: la>0.0, lambda al: al>0.0, lambda la: la>0.0, lambda la: la>0.0]

#%%
"Bucle para realizar los histogramas para cada planta"
for N in range(1,37): #N es el número de planta
    modelo = PHSmodel(init_dict)
    datos_trigo = datosendic(oxigeno = [20, 210], N = N) 
    modelo.uploadExpData(datos_trigo)
    #Filtra los datos quitando los primeros 5
    datos1=modelo.dataAt(constvar='Qin',oxigeno=20)
    datos2=modelo.dataAt(constvar='Qin',oxigeno=210)
    datos3=modelo.dataAt(constvar='Ci',oxigeno=20)
    datos4=modelo.dataAt(constvar='Ci',oxigeno=210)
    #Agrega identificadores para los niveles de oxígeno
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
    ### Define the Forward map with signature F( theta, Data)
    def F(theta, Data):
        gmo, Vcmax, delta, Tp, thetaNPR, thetaPR, JmaxNPR, JmaxPR, k2_LLNPR, k2_LLPR, Sco, Kmo, Kmc, Rd_NPR, Rd_PR = theta
        P = Data
        A = modelo.fABayes(P, gmo, Vcmax, delta, Tp, thetaNPR, thetaPR, JmaxNPR, JmaxPR, k2_LLNPR, k2_LLPR, Sco, Kmo, Kmc, Rd_NPR, Rd_PR)
        return A
    
    #Define el error
    sigma = np.array(dat['Error'])
    
    #%%
    #Cuantificiación de incertidumbre bayesiana
    buq = BUQ( q = q, data = dat["A"], logdensity=logdensity, sigma=sigma,\
        F = F, t = Data, par_names = par_names, par_prior = par_prior, \
        par_supp=par_supp)

    #Leer los resultados del MCMC (Cuando ya se hayan hecho los MCMC)
    buq.LoadtwalkOutput(f'MCMC{N}.csv1') #Leo el MCMC ya guardado
    #%%

    "Histogramas de los parámetros"
    #Estilo y tamaños de fuentes
    rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12})
    fig, axes = subplots(nrows=3, ncols=5, figsize=(15, 10)) #Cuadricula de 5x3 para hacer los histogramas juntos
    axes = axes.flatten()  #Para poder indexarlos linealmente

    "Gráfica de los histogramas"
    #Bucle para graficar cada histograma
    for k in range(15): #Son 15 parámetros
        ax = axes[k] #Llama a la k-ésima entrada de la cuadricula
        buq.PlotPost(par=k, bins=15, burn_in=1_000_000, ax=ax) #Histograma graficado en la entrada k de la cuadricula
        # Agregar letra del inciso
        letra = chr(97 + k)  # 97 es el código ASCII de 'a'
        ax.set_title(f"({letra})", loc='center') #Pone el inciso a cada histograma
        if N!=17:
           df = pd.read_excel('./Parametros_ref.xlsx', header=[0]) #Leo el docuemnto con los valores estimados por Yin et. al. (2009)
           x_ref = df[f"Planta {N}"]  #Uso los valores de referencia de los parámetros para cada planta
           ax.plot(x_ref[k], 0, 'r*', markersize=15, transform=ax.get_xaxis_transform()) #Agrego un astérisco con el valor de referencia a cada histograma
           
    #Borrar los ejes sobrantes
    for j in range(k+1, len(axes)):
        fig.delaxes(axes[j]) 
    tight_layout()
    
    #Guarda la imagen en la carpeta y con el nombre para cada planta
    fig.savefig("Histogramas/Diseño convencional/{plant_names[N]}.png", dpi=300, bbox_inches='tight') #Guarda la imagen con los histogramas

