"""
Este código es para hacer los MCMC para las 36 plantas de forma paralela
"""

from concurrent.futures import ProcessPoolExecutor
import numpy as np
from pytwalk import BUQ
from FvCB_V11 import PHSmodel
from DataDict import init_dict
from readdata import *
from scipy.stats import norm, uniform

"Nombres de los parámetros"
par_names = ["gmo", "Vcmax", "delta", "Tp", r"$\theta_{NPR}$", r"$\theta_{PR}$", r"$J_{max}^{NPR}$",
             r"$J_{max}^{PR}$", r"$k2_{LL}^{NPR}$", r"$k2_{LL}^{PR}$", "Sco", "Kmo", "Kmc", r"$Rd_{NPR}$", r"$Rd_{PR}$"]


"A prioris de los parámetros"
par_prior = [
    uniform(0.0, 0.001), uniform(10.0, 190), uniform(0.0, 4.0), uniform(0.0, 30),
    uniform(0.0, 1.0), uniform(0.0, 1.0), uniform(2.0, 498), uniform(2.0, 498),
    uniform(0.1, 0.8), uniform(0.1, 0.8), uniform(0.0, 5.0), uniform(1.0, 69),
    uniform(1.0, 69), uniform(0.0, 10), uniform(0.0, 10)
]

"Soportes de las a priori"
par_supp = [lambda x: x > 0] * 15

def run_N(N):
    modelo = PHSmodel(init_dict)
    datos = datosendic(oxigeno=[20, 210], N=N)
    modelo.uploadExpData(datos)
    dat = modelo.dataAll()
    Data = np.array([dat['I'], dat['C'], dat['O'], dat['T'], dat['Osp']])
    sigma = np.array(dat['Error'])
    
    def F(theta, Data):
        gmo, Vcmax, delta, Tp, thetaNPR, thetaPR, JmaxNPR, JmaxPR, k2_LLNPR, k2_LLPR, Sco, Kmo, Kmc, Rd_NPR, Rd_PR = theta
        P = Data
        A = modelo.fABayes(P, gmo, Vcmax, delta, Tp, thetaNPR, thetaPR, JmaxNPR, JmaxPR, k2_LLNPR, k2_LLPR, Sco, Kmo, Kmc, Rd_NPR, Rd_PR)
        return A
    q = 15
    logdensity = norm.logpdf
    simdata = lambda n, loc, scale: norm.rvs(size=n, loc=loc, scale=scale)

    buq = BUQ(q=q, data=dat["A"], logdensity=logdensity, simdata=simdata, sigma=sigma,
              F=F, t=Data, par_names=par_names, par_prior=par_prior, par_supp=par_supp)
    buq.RunMCMC(T=5_000_000, burn_in=0, fnam=f"buq_NC{N}.csv1")

if __name__ == "__main__":
    with ProcessPoolExecutor() as executor:
        executor.map(run_N, range(1, 37))
