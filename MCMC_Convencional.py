"""
Hace los MCMC con datos sinteticos usando el diseno convencional
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
simdata = lambda n, loc, scale: norm.rvs( size=n, loc=loc, scale=scale) #Simula datos con la distribucion normal
#Nombres de los parametros
par_names=["gmo", "Vcmax", "delta", "Tp", r"$\theta_{NPR}$",
            r"$\theta_{PR}$", r"$J_{max}^{NPR}$", 
            r"$J_{max}^{PR}$", r"$k2_{LL}^{NPR}$" , 
            r"$k2_{LL}^{PR}$", "Sco", "Kmo", "Kmc", 
            r"$Rd_{NPR}$", r"$Rd_{PR}$"]
               
#VarsOrder =  gmo, Vcmax, delta, Tp, thetaNPR, thetaPR, JmaxNPR, JmaxPR, k2_LLNPR, k2_LLPR, Sco, Kmo, Kmc, Rd_NPR, Rd_PR 

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
"Funcion para correr los MCMC para cada planta"
def run_N(N): #N es el numero de planta
    modelo = PHSmodel(init_dict)
    datos_trigo = datosendic(oxigeno = [20, 210], N = N) 
    modelo.uploadExpData(datos_trigo)
    #Filtra los datos quitando las primeras 5 
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

    ### Define the Forward map with signature F( theta, Data)
    def F(theta, Data):
        gmo, Vcmax, delta, Tp, thetaNPR, thetaPR, JmaxNPR, JmaxPR, k2_LLNPR, k2_LLPR, Sco, Kmo, Kmc, Rd_NPR, Rd_PR = theta
        P = Data
        A = modelo.fABayes(P, gmo, Vcmax, delta, Tp, thetaNPR, thetaPR, JmaxNPR, JmaxPR, k2_LLNPR, k2_LLPR, Sco, Kmo, Kmc, Rd_NPR, Rd_PR)
        return A

    #Error
    sigma = np.array(dat['Error']) 
    
    #%%
    #Cuantificacion de incertidumbre bayesiana (sin datos)
    buq = BUQ( q = q, data = None, logdensity=logdensity,simdata=simdata, sigma=sigma,\
         F = F, t = Data, par_names = par_names, par_prior = par_prior, par_supp=par_supp)
        
    #%%
    "Simulacion de datos sinteticos"
    #Valores de referencia para los parametros
    df = pd.read_excel('./Parametros_ref.xlsx', header=[0]) #Lee el excel con los valores de referencia
    x_ref = df[f"Planta {N}"]  #Toma el valor de referencia para la planta N (Yin et al p.41-44)
    
    sam_size = buq.F(theta=x_ref,Data=buq.t).shape[0] #Tamano de la muestra a simular
    data = buq.simdata(n=sam_size,loc=buq.F(theta=x_ref,Data=buq.t),scale=buq.sig) #Datos simulados
    
    #Cuantificacion de incertidumbre bayesiana (con datos simulados)
    buq = BUQ( q = q, data = data, logdensity=logdensity,simdata=simdata, sigma=sigma,\
         F = F, t = Data, par_names = par_names, par_prior = par_prior, par_supp=par_supp)
    
    #Correr el MCMC y lo guarda como archivo 'MCMC_conv{N}.csv1'
    buq.RunMCMC( T = 5_000_000,fnam=f"MCMC_conv{N}.csv1") 
    
#%%
"Ejecuta los MCMC en paralelo para las N (1-36) plantas"
if __name__ == "__main__":
    with ProcessPoolExecutor() as executor:
        executor.map(run_N, (x for x in range(1, 37) if x != 17))

