"Hace los MCMC con valores fijos de V_{cmax}"

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
from matplotlib.pylab import subplots, plot, close, hist, show, axvline, title, tight_layout, axvline, scatter, rcParams
from matplotlib.patches import Patch

#%%
"Configuraci�n para el dise�o de zigzag"
Czz = np.linspace(1300,0,61) #CO2
Izz = np.full(61,2000) #luz incidente
for i in range(10):
    for j in [0,20,40]:
        Izz[i+j] = 2000 - i*200
for i in range(10,20):
    for j in [0,20,40]:
        Izz[i+j] = (i-10)*200

#Error para el dise�o de zigzag
zigzag = pd.read_excel('./Zigzag.xlsx', header=[0]) #Lee el excel con los errores para el dise�o de zigzag
sigma_zz = zigzag["Error"]

#%%
"Configuaci�n para el dise�o de rejilla"     
Igrid = outer( linspace( 0, 2000, num=20), ones(20)) #Luz incidente
Cgrid = outer( linspace( 0., 1300, num=20), ones(20)).T #Niveles de CO2
Igrid = np.array([Igrid.ravel(),Igrid.ravel()]).ravel()
Cgrid = np.array([Cgrid.ravel(),Cgrid.ravel()]).ravel()

#Error para el dise�o de rejilla
grid = pd.read_excel('./Grid.xlsx', header=[0]) #Lee el excel con los errores para el dise�o de rejilla
sigma_grid = grid["Error"]

#%%
q = 15 # Numero de par�metros a inferir
logdensity=norm.logpdf #log-densidad del modelo
simdata = lambda n, loc, scale: norm.rvs( size=n, loc=loc, scale=scale) #Simula datos con la distribuci�n normal
#Nombres de los par�metros
par_names=["gmo", "Vcmax", "delta", "Tp", r"$\theta_{NPR}$",
            r"$\theta_{PR}$", r"$J_{max}^{NPR}$", 
            r"$J_{max}^{PR}$", r"$k2_{LL}^{NPR}$" , 
            r"$k2_{LL}^{PR}$", "Sco", "Kmo", "Kmc", 
            r"$Rd_{NPR}$", r"$Rd_{PR}$"]

#VarsOrder =  gmo, Vcmax, delta, Tp, thetaNPR, thetaPR, JmaxNPR, JmaxPR, k2_LLNPR, k2_LLPR, Sco, Kmo, Kmc, Rd_NPR, Rd_PR 

#A prioris de los par�metros
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
"Dise�os para el MCMC"
dVcmax = [] #Lista para guardar lo dise�os
for N in range(1,37): #Lista de plantas
    if N != 17:
        for val in [30,35,50,55,100,150]: #Lista de valores de V_{cmax}
            df = pd.read_excel('./Parametros_ref.xlsx', header=[0]) #Lee el excel con los valores de referencia
            x_ref = df[f"Planta {N}"]  #Toma los valores de referencia para la planta N (Yin et al p.41-44)
            V = x_ref.copy() #Crea una copia de los valores de referencia
            V[1] = val #Cambia el valor de V_{cmax} por los de la lista
            #Agregar a la lista (Valores de referencia, dise�o, n�mero de planta)
            dVcmax.append((V, "zz", N)) #Zigzag
            dVcmax.append((V, "conv", N)) #Convencional
            dVcmax.append((V, "grid", N)) #Rejilla


#%%
"Funci�n para correr los MCMC para cada valor de V_{cmax}, dise�o y planta"
def run_single_task(task):
    V, d_exp, N = task #Valores de referencia, dise�o y n�mero de planta
    i = V[1] #Valor de V_{cmax} 
    modelo = PHSmodel(init_dict)
    datos_trigo = datosendic(oxigeno = [20, 210], N = N)
    modelo.uploadExpData(datos_trigo)
    
    #%%
    "Configuraci�n para el dise�o convencional"
    #Filtra los datos quitando las primeras 5
    datos1=modelo.dataAt(constvar='Qin',oxigeno=20)
    datos2=modelo.dataAt(constvar='Qin',oxigeno=210)
    datos3=modelo.dataAt(constvar='Ci',oxigeno=20)
    datos4=modelo.dataAt(constvar='Ci',oxigeno=210)
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
    
    #Error
    sigma = np.array(dat['Error'])

    #%%
    "Configuraci�n para el diseño de zigzag"
    dat = modelo.dataAll()
    
    #Temperatura de la hoja
    T_NPR = np.full(61,np.mean(datos_trigo["e1"]["Tleaf"]))
    T_PR = np.full(61,np.mean(datos_trigo["e2"]["Tleaf"]))

    #Oxígeno
    O_NPR = np.full(61,np.mean(datos_trigo["e1"]["PO2"]))
    O_PR = np.full(61,np.mean(datos_trigo["e2"]["PO2"]))

    #Niveles de oxígeno
    Osp_NPR = np.full(61,20)
    OSP_PR = np.full(61,210)
    
    # Set-up experimental data (zigzag)
    Data_zz = np.array([np.array([Izz,Izz]).ravel(),
                        np.array([Czz,Czz]).ravel(),
                        np.array([T_NPR,T_PR]).ravel(),
                        np.array([O_NPR,O_PR]).ravel(),
                        np.array([Osp_NPR,OSP_PR]).ravel()])
    
    #%%
    "Configuraci�n para el diseño de rejilla"
    # Set-up experimental data
    mask = dat['Osp'] == 20 #Filtrar los datos de oxígeno bajo
    #Niveles de oxigeno
    O = np.array([np.full(int(len(Cgrid)/2),np.mean(dat['O'][mask])),np.full(int(len(Cgrid)/2),np.mean(dat['O'][~mask]))]).ravel()
    #Temperatura
    T = np.array(np.full(len(Cgrid),np.mean(dat['T']))).ravel()
    #Identificador de nivel de ox�geno
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
    #MCMC para el dise�o de zigzag
    if d_exp == "zz":
        #Cuantificaci�n de incertidumbre bayesiana (sin datos)
        buqzz_i = BUQ(q=q, data=None, logdensity=logdensity, simdata=simdata, sigma=sigma_zz,
                      F=F, t=Data_zz, par_names=par_names, par_prior=par_prior, par_supp=par_supp)
        #Tama�o de muestra a simular
        sam_size_zz = buqzz_i.F(theta=x_ref,Data=buqzz_i.t).shape[0]
        #Datos simulados
        datazz_i = buqzz_i.simdata(n=sam_size_zz,loc=buqzz_i.F(theta=x_ref,Data=buqzz_i.t),scale=buqzz_i.sig)
        #Cuantificaci�n de incertidumbre bayesiana (con datos simulados)
        buqzz_i = BUQ( q = q, data = datazz_i, logdensity=logdensity,simdata=simdata, sigma=sigma_zz,\
            F = F, t = Data_zz, par_names = par_names, par_prior = par_prior, par_supp=par_supp)
        #Correr el MCMC y lo guarda como archivo .csv1
        buqzz_i.RunMCMC(T=5_000_000, burn_in=0, fnam=f"MCMCzigzag_V{i}_{N}.csv1")
        
    #%%
    #MCMC para el dise�o convencional
    elif d_exp == "conv":
        #Cuantificaci�n de incertidumbre bayesiana (sin datos)
        buq_i = BUQ(q=q, data=None, logdensity=logdensity, simdata=simdata, sigma=sigma,
                    F=F, t=Data, par_names=par_names, par_prior=par_prior, par_supp=par_supp)
        #Tama�o de muestra a simular
        sam_size = buq_i.F(theta=x_ref,Data=buq_i.t).shape[0] 
        #Datos simulados
        data=buq_i.simdata(n=sam_size,loc=buq_i.F(theta=x_ref,Data=buq_i.t),scale=buq_i.sig) 
        #Cuantificaci�n de incertidumbre bayesiana (con datos simulados)
        buq_i = BUQ( q = q, data = data, logdensity=logdensity,simdata=simdata, sigma=sigma,\
             F = F, t = Data, par_names = par_names, par_prior = par_prior, par_supp=par_supp)
        #Correr el MCMC y lo guarda como archivo .csv1
        buq_i.RunMCMC(T=5_000_000, burn_in=0, fnam=f"MCMCconv_V{i}_{N}.csv1")
        
    #%%
    #MCMC para el dise�o de rejilla
    elif d_exp == "grid":
        #Cuantificaci�n de incertidumbre bayesiana (sin datos)
        buqgrid_i = BUQ(q=q, data=None, logdensity=logdensity, simdata=simdata, sigma=sigma_grid,
                        F=F, t=Data_grid, par_names=par_names, par_prior=par_prior, par_supp=par_supp)
        #Tama�o de muestra a simular
        sam_size_grid = buqgrid_i.F(theta=x_ref,Data=buqgrid_i.t).shape[0]
        #Datos simulados
        data_grid = buqgrid_i.simdata(n=sam_size_grid,loc=buqgrid_i.F(theta=x_ref,Data=buqgrid_i.t),scale=buqgrid_i.sig)
        #Cuantificaci�n de incertidumbre bayesiana (con datos simulados)
        buqgrid_i = BUQ( q = q, data = data_grid, logdensity=logdensity,simdata=simdata, sigma=sigma_grid,\
            F = F, t = Data_grid, par_names = par_names, par_prior = par_prior, par_supp=par_supp)
        #Correr el MCMC y lo guarda como archivo .csv1
        buqgrid_i.RunMCMC(T=5_000_000, burn_in=0, fnam=f"MCMCgrid_V{i}_{N}.csv1")
        
#%%
"Ejecuta los MCMC en paralelo para las distintos dise�os del experimento"
if __name__ == "__main__":
    with ProcessPoolExecutor() as executor:
        executor.map(run_single_task, dVcmax)

