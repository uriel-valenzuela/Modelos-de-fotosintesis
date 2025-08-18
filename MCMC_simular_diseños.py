"""
Hace la inferencia para los diseños de rejilla y zigzag con datos sintéticos
"""
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from pytwalk import BUQ
from FvCB_V11 import PHSmodel
from DataDict import init_dict
from readdata import *
from scipy.stats import bernoulli, weibull_min, norm, gamma, expon, uniform
from numpy import loadtxt, linspace, outer, ones
from matplotlib.pylab import subplots, plot, close, hist, show, axvline, title, tight_layout, axvline, scatter

"Datos para el diseño de zigzag"

Czz = np.linspace(1300,0,61) #CO2
Izz = np.full(61,2000) #luz incidente
#Zigzag
for i in range(10):
    for j in [0,20,40]:
        Izz[i+j] = 2000 - i*200
for i in range(10,20):
    for j in [0,20,40]:
        Izz[i+j] = (i-10)*200
zigzag = pd.read_excel('./Zigzag.xlsx', header=[0]) #Lee el excel con los errores para el dis de zz
sigma_zz = zigzag["Error"]
Igrid = outer( linspace( 0, 2000, num=20), ones(20)) #Luz incidente
Cgrid = outer( linspace( 0., 1300, num=20), ones(20)).T #Niveles de CO2
#Rejilla
Igrid = np.array([Igrid.ravel(),Igrid.ravel()]).ravel() 
Cgrid = np.array([Cgrid.ravel(),Cgrid.ravel()]).ravel()

grid = pd.read_excel('./Grid.xlsx', header=[0]) #Lee el excel con los errores para el diseño de rejilla
sigma_grid = grid["Error"]

def run_N (N): 
    if N!=17:
        modelo = PHSmodel(init_dict)
        # para la planta N = 2
        datos_trigo = datosendic(oxigeno = [20, 210], N = 2)
        modelo.uploadExpData(datos_trigo)
        dat = modelo.dataAll()

        #%%
        # Set-up experimental data
        #Temperatura de la hoja
        T_NPR = np.full(61,np.mean(datos_trigo["e1"]["Tleaf"]))
        T_PR = np.full(61,np.mean(datos_trigo["e2"]["Tleaf"]))
        #Oxígeno
        O_NPR = np.full(61,np.mean(datos_trigo["e1"]["PO2"]))
        O_PR = np.full(61,np.mean(datos_trigo["e2"]["PO2"]))
        #Niveles de oxígeno
        Osp_NPR = np.full(61,20)
        OSP_PR = np.full(61,210)
        Data_zz = np.array([np.array([Izz,Izz]).ravel(),
                            np.array([Czz,Czz]).ravel(),
                            np.array([O_NPR,O_PR]).ravel(),
                            np.array([T_NPR,T_PR]).ravel(),
                            np.array([Osp_NPR,OSP_PR]).ravel()])
        #%%
        "Datos para el diseño de rejilla"
        # Set-up experimental data
        mask = dat['Osp'] == 20 #Filtrar los datos de oxígeno bajo
        O = np.array([np.full(int(len(Cgrid)/2),np.mean(dat['O'][mask])),np.full(int(len(Cgrid)/2),np.mean(dat['O'][~mask]))]).ravel()
        T = np.array(np.full(len(Cgrid),np.mean(dat['T']))).ravel()
        Osp = np.array([np.full(int(len(Cgrid)/2), 20),np.full(int(len(Cgrid)/2),210)]).ravel()
        print("Esta haciendo algo2")
        Data_grid = np.array([Igrid, Cgrid, O, T, Osp])
        print("hizo los setups")
        #%%
        ### Define the Forward map with signature F( theta, Data)
        def F(theta, Data):
            gmo, Vcmax, delta, Tp, thetaNPR, thetaPR, JmaxNPR, JmaxPR, k2_LLNPR, k2_LLPR, Sco, Kmo, Kmc, Rd_NPR, Rd_PR = theta
            P = Data
            A = modelo.fABayes(P, gmo, Vcmax, delta, Tp, thetaNPR, thetaPR, JmaxNPR, JmaxPR, k2_LLNPR, k2_LLPR, Sco, Kmo, Kmc, Rd_NPR, Rd_PR)
            return A
        q = 15 # Numero de parámetros a inferir
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
        par_supp  = [ lambda al: al>0.0, lambda la: la>0.0, lambda al: al>0.0, lambda la: la>0.0, lambda al: al>0.0,
                      lambda al: al>0.0, lambda la: la>0.0, lambda al: al>0.0, lambda la: la>0.0, lambda al: al>0.0,
                      lambda al: al>0.0, lambda la: la>0.0, lambda al: al>0.0, lambda la: la>0.0, lambda la: la>0.0]
        # %%
        print(f"hizo todo los datos antes de buqs{N}")
        #Diseño de rejilla
        buqgrid = BUQ( q = q, data = None, logdensity=logdensity,simdata=simdata, sigma=sigma_grid,\
            F = F, t = Data_grid, par_names = par_names, par_prior = par_prior, par_supp=par_supp)
        #Diseño de zigzag
        buqzigzag = BUQ( q = q, data = None, logdensity=logdensity,simdata=simdata, sigma=sigma_zz,\
            F = F, t = Data_zz, par_names = par_names, par_prior = par_prior, par_supp=par_supp)
        # %%
        print(f"hizo buqs {N}")
        "Simulación de datos sintéticos"
        #Valores de referencia para los parámetros
        df = pd.read_excel('./Parametros_ref.xlsx', header=[0])
        x_ref = df[f"Planta {N}"]  #Yin et al p.41-44
        #Valores sintéticos
        #buqzigzag.SimData(x = x_ref)
        #buqgrid.SimData(x = x_ref)
        sam_size_zz = buqzigzag.F(theta=x_ref,Data=buqzigzag.t).shape[0]
        buqzigzag.data=buqzigzag.simdata(n=sam_size_zz,loc=buqzigzag.F(theta=x_ref,Data=buqzigzag.t),scale=buqzigzag.sig)
        sam_size_grid = buqgrid.F(theta=x_ref,Data=buqgrid.t).shape[0]
        buqgrid.data=buqgrid.simdata(n=sam_size_grid,loc=buqgrid.F(theta=x_ref,Data=buqgrid.t),scale=buqgrid.sig)
        print(f"hizos buqsim{N}")
        buqgrid.RunMCMC( T= 5_000_000, burn_in=0, fnam=f"simbuqgrid{N}.csv1") #MCMC
        buqzigzag.RunMCMC( T= 5_000_000, burn_in=0, fnam=f"simbuqzigzag{N}.csv1") #MCMC
        ##%%




if __name__ == "__main__":
    with ProcessPoolExecutor() as executor:
        executor.map(run_N, range(1, 37))
