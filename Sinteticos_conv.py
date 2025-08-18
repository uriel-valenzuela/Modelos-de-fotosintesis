"""
Código para hacer la inferencia con datos sintéticos y el diseño convencional
"""

from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from pytwalk import BUQ
from FvCB_V11 import PHSmodel
from DataDict import init_dict
from readdata import *
from scipy.stats import bernoulli, weibull_min, norm, gamma, expon, uniform
from matplotlib.pylab import subplots, plot, close, hist, show,rcParams, axvline, title, scatter, fill_between, tight_layout


"Inferencia bayesiana para el diseño convencional"
q = 15 # Numero de parnmetros a inferir
logdensity=norm.logpdf
simdata = lambda n, loc, scale: norm.rvs( size=n, loc=loc, scale=scale)
par_names=["gmo", "Vcmax", "delta", "Tp", r"$\theta_{NPR}$",
            r"$\theta_{PR}$", r"$J_{max}^{NPR}$", 
            r"$J_{max}^{PR}$", r"$k2_{LL}^{NPR}$" , 
            r"$k2_{LL}^{PR}$", "Sco", "Kmo", "Kmc", 
            r"$Rd_{NPR}$", r"$Rd_{PR}$"]
               
#VarsOrder =  gmo, Vcmax, delta, Tp, thetaNPR, thetaPR, JmaxNPR, JmaxPR, k2_LLNPR, k2_LLPR, Sco, Kmo, Kmc, Rd_NPR, Rd_PR 

par_prior =[uniform(0.0, scale=0.001),    #gmo,
            uniform(10.0,scale=190),      #Vcmax
            uniform(0.0, scale=4.0),      #delta
            uniform(0.0, scale=30),      #Tp 
            uniform(0.0, scale=1.0),     #ThetaNPR
            uniform(0.0, scale=1.0),     #ThetaPR
            uniform(2.0, scale=498),     #JmaxNPR
            uniform(2.0, scale=498),     #JmaxPR,
            uniform(0.1, scale=0.8),     #k2_LLNPR,
            uniform(0.1, scale=0.8),     #k2_LL_PR,
            uniform(0.0, scale=5.0),     #Sco
            uniform(1.0, scale=69),      #Kmc, limites en Pa 
            uniform(1.0, scale=69),      #Kmo, limites en kPa 
            uniform(0.0, scale=10),      #Rd_NPR
            uniform(0.0, scale=10)      #Rd_PR 
            ] 

def run_N(N):
    #N = 2 #Planta que se quiere usar 1-36
    modelo = PHSmodel(init_dict)
    # para la planta N
    datos_trigo = datosendic(oxigeno = [20, 210], N = N) 
    modelo.uploadExpData(datos_trigo)
    #Quitar primeros 5 datos 
    datos1=modelo.dataAt(constvar='Qin',oxigeno=20)
    datos2=modelo.dataAt(constvar='Qin',oxigeno=210)
    datos3=modelo.dataAt(constvar='Ci',oxigeno=20)
    datos4=modelo.dataAt(constvar='Ci',oxigeno=210)
    datos1['Osp'] = np.full(len(datos1["A"]),20)
    datos2['Osp'] = np.full(len(datos2["A"]),210)
    datos3['Osp'] = np.full(len(datos3["A"]),20)
    datos4['Osp'] = np.full(len(datos4["A"]),210)
    
    
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

    #Soporte de las a priori
    par_supp  = [ lambda al: al>0.0, lambda la: la>0.0, lambda al: al>0.0, lambda la: la>0.0, lambda al: al>0.0,
                  lambda al: al>0.0, lambda la: la>0.0, lambda al: al>0.0, lambda la: la>0.0, lambda al: al>0.0,
                  lambda al: al>0.0, lambda la: la>0.0, lambda al: al>0.0, lambda la: la>0.0, lambda la: la>0.0]
    #Cuantificación de incertidumbre bayesiana
    buq = BUQ( q = q, data = None, logdensity=logdensity,simdata=simdata, sigma=sigma,\
         F = F, t = Data, par_names = par_names, par_prior = par_prior, par_supp=par_supp)
        
    "Simulación de datos sintéticos"
    #Valores de referencia para los parámetros
    df = pd.read_excel('./Parametros_ref.xlsx', header=[0]) #Lee el excel con los valores de referencia
    x_ref = df[f"Planta {N}"]  #Toma el valor de referencia para la planta N (Yin et al p.41-44)
    
    #buq.SimData(x = x_ref) #Simula los datos sintéticos (Esto hay que cambiarlo)

    sam_size = buq.F(theta=x_ref,Data=buq.t).shape[0]
    data=buq.simdata(n=sam_size,loc=buq.F(theta=x_ref,Data=buq.t),scale=buq.sig)
    buq = BUQ( q = q, data = data, logdensity=logdensity,simdata=simdata, sigma=sigma,\
         F = F, t = Data, par_names = par_names, par_prior = par_prior, par_supp=par_supp)
    buq.RunMCMC( T = 5_000_000,fnam=f"sMCMC_conv{N}.csv1") #Hace el MCMC y lo guarda (Hacer solo una vez)
    #buq.LoadtwalkOutput(f'sMCMC_conv{N}.csv1') #Leo el MCMC ya guardado 
    #%%

    "Histogramas de los parámetros"
    rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12})
    fig, axes = subplots(nrows=5, ncols=3, figsize=(12, 15))
    axes = axes.flatten()  # para poder indexarlos

    for k in range(15):
        ax = axes[k]
        buq.PlotPost(par=k, bins=15, burn_in=1_000_000, ax=ax)
        # Agregar letra del inciso
        letra = chr(97 + k)  # 97 es el código ASCII de 'a'
        ax.set_title(f"({letra})", loc='center')

    for j in range(k+1, len(axes)):
        fig.delaxes(axes[j])
    tight_layout()
    fig.savefig(f"shist_conv{N}.png", dpi=300, bbox_inches='tight')
    print(f'Debió imprimir {N}')

#(Habilitar esto si se quiere hacer el MCMC por primera vez y cambiar el for por def run_N(N))
if __name__ == "__main__":
    with ProcessPoolExecutor() as executor:
        executor.map(run_N, (x for x in range(1, 37) if x != 17))

