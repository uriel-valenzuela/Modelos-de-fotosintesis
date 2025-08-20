#%%
import numpy as np
from pytwalk import BUQ
from FvCB_V11 import PHSmodel
from DataDict import init_dict
from readdata import *
from scipy.stats import bernoulli, weibull_min, norm, gamma, expon, uniform


modelo = PHSmodel(init_dict)
# para la planta N = 2
datos_trigo = datosendic(oxigeno = [20, 210], N = 2)
modelo.uploadExpData(datos_trigo)
dat = modelo.dataAll()
# Set-up experimental data
Data = np.array([dat['I'], dat['PCi'], dat['O'], dat['T'], dat['Osp']]) 

### Define the Forward map with signature F( theta, Data)
def F(theta, Data):
    gmo, Vcmax, delta, Tp, thetaNPR, thetaPR, JmaxNPR, JmaxPR, k2_LLNPR, k2_LLPR, Sco, Kmo, Kmc, Rd_NPR, Rd_PR = theta
    P = Data
    A = modelo.fABayes(P, gmo, Vcmax, delta, Tp, thetaNPR, thetaPR, JmaxNPR, JmaxPR, k2_LLNPR, k2_LLPR, Sco, Kmo, Kmc, Rd_NPR, Rd_PR)
    return A

#sigma = 0.8
sigma = np.array(dat['Error'])
# sigma = np.array([error A_1 ])
### logdensity: log of the density of G, with signature logdensity( data, loc, scale)
### see docstring of BUQ
q = 15 # Numero de parámetros a inferir
logdensity=norm.logpdf
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
            uniform(1.0, scale=49),      #Kmc, limites en Pa 
            uniform(1.0, scale=49),      #Kmo, limites en kPa 
            uniform(0.0, scale=10),      #Rd_NPR
            uniform(0.0, scale=10)      #Rd_PR 
            ] 

par_supp  = [ lambda al: al>0.0, lambda la: la>0.0, lambda al: al>0.0, lambda la: la>0.0, lambda al: al>0.0,
              lambda al: al>0.0, lambda la: la>0.0, lambda al: al>0.0, lambda la: la>0.0, lambda al: al>0.0,
              lambda al: al>0.0, lambda la: la>0.0, lambda al: al>0.0, lambda la: la>0.0, lambda la: la>0.0]

buq = BUQ( q = q, data = None, logdensity=logdensity, sigma=sigma,\
    F = F, t = Data, par_names = par_names, par_prior = par_prior, \
    par_supp=par_supp)

buq.SimData(x = [1,2,3,...])
buq.RunMCMC( T = 50_000, burn_in = 0, fnam = 'Ejemplo.csv')
#%%
#buq.LoadtwalkOuput(fnam='Corrida3.csv')
buq.Ana(burn_in=10_000)
#%%

for k in range(0,15):
    buq.PlotPost(par=k, bins = 15, burn_in=10_000)

# %%

# %%
