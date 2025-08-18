"""
Código para hacer la inferencia bayesiana y obtener estimadores a posterior de A junto a 
intervalos de confiabilidad bayesiana.
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
logdensity=norm.logpdf #log-densidad del modelo
#Nombres de los parámetros
par_names=["gmo", "Vcmax", "delta", "Tp", r"$\theta_{NPR}$",
            r"$\theta_{PR}$", r"$J_{max}^{NPR}$", 
            r"$J_{max}^{PR}$", r"$k2_{LL}^{NPR}$" , 
            r"$k2_{LL}^{PR}$", "Sco", "Kmo", "Kmc", 
            r"$Rd_{NPR}$", r"$Rd_{PR}$"]
               
#VarsOrder =  gmo, Vcmax, delta, Tp, thetaNPR, thetaPR, JmaxNPR, JmaxPR, k2_LLNPR, k2_LLPR, Sco, Kmo, Kmc, Rd_NPR, Rd_PR 

#A prioris de los parámetros
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

for N in range(1,37):
    #N  #Planta que se quiere usar 1-36
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
    for key in datos1
}
    
    # Set-up experimental data
    Data = np.array([dat['I'], dat['PCi'], dat['O'], dat['T'], dat['Osp']]) 
    print(len(dat['I']))
    ### Define the Forward map with signature F( theta, Data)
    def F(theta, Data):
        gmo, Vcmax, delta, Tp, thetaNPR, thetaPR, JmaxNPR, JmaxPR, k2_LLNPR, k2_LLPR, Sco, Kmo, Kmc, Rd_NPR, Rd_PR = theta
        P = Data
        A = modelo.fABayes(P, gmo, Vcmax, delta, Tp, thetaNPR, thetaPR, JmaxNPR, JmaxPR, k2_LLNPR, k2_LLPR, Sco, Kmo, Kmc, Rd_NPR, Rd_PR)
        return A

    sigma = np.array(dat['Error'])
    # sigma = np.array([error A_1 ])
    ### logdensity: log of the density of G, with signature logdensity( data, loc, scale)
    ### see docstring of BUQ

    #Soportes de las a priori
    par_supp  = [ lambda al: al>0.0, lambda la: la>0.0, lambda al: al>0.0, lambda la: la>0.0, lambda al: al>0.0,
                  lambda al: al>0.0, lambda la: la>0.0, lambda al: al>0.0, lambda la: la>0.0, lambda al: al>0.0,
                  lambda al: al>0.0, lambda la: la>0.0, lambda al: al>0.0, lambda la: la>0.0, lambda la: la>0.0]

    #Cuantificiación de incertidumbre bayesiana
    buq = BUQ( q = q, data = dat["A"], logdensity=logdensity, sigma=sigma,\
        F = F, t = Data, par_names = par_names, par_prior = par_prior, \
        par_supp=par_supp)

    #Correr el MCMC (Hacerlo la primera vez)
    #buq.RunMCMC( T = 5_000_000,fnam=f"MCMC{N}.csv1")
    #%%
    #Leer los resultados del MCMC (Cuando ya se hayan hecho los MCMC)
    buq.LoadtwalkOutput(f'MCMC{N}.csv1') #Leo el MCMC ya guardado
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
    fig, axes = subplots(nrows=3, ncols=5, figsize=(15, 10))
    axes = axes.flatten()  # para poder indexarlos

    for k in range(15):
        ax = axes[k]
        buq.PlotPost(par=k, bins=15, burn_in=1_000_000, ax=ax)
        # Agregar letra del inciso
        letra = chr(97 + k)  # 97 es el código ASCII de 'a'
        ax.set_title(f"({letra})", loc='center')
        if N!=17:
           df = pd.read_excel('./Parametros_ref.xlsx', header=[0])
           x_ref = df[f"Planta {N}"]  
           ax.plot(x_ref[k], 0, 'r*', markersize=15, transform=ax.get_xaxis_transform())

    for j in range(k+1, len(axes)):
        fig.delaxes(axes[j])

    tight_layout()
    fig.savefig("hist{N}.png", dpi=300, bbox_inches='tight')



  # %%
    "Valores a posterior de A "
    """
    A_post = []
    for k in range(1_000_000,len(buq.Output[:,0])):
        A_post.append(F(buq.Output[k,:15],Data))
    
    df = pd.DataFrame(A_post)
    df.to_csv(f'Apost{N}.csv', index=False, header=False) #Lo guardo (Si no lo tengo aún)
    """
    A_post = pd.read_csv(f'ApostC{N}.csv',header=None) #Lo leo (si ya lo guardé)

    "Datos observados"
    rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12})
    fig, axs = subplots(1, 2, figsize=(15, 7))

    axs[0].scatter(datos1["C"],datos1["A"], label=f"O$_{{2}}=2\\%$",color="blue")
    axs[0].scatter(datos2["C"],datos2["A"],label=f"O$_{{2}}=21\\%$",color="green")
    axs[1].scatter(datos3["I"],datos3["A"],label=f"O$_{{2}}=2\\%$",color="blue")
    axs[1].scatter(datos4["I"],datos4["A"],label=f"O$_{{2}}=21\\%$",color="green")
    axs[0].set_xlabel(f"C$_{{\\text{{i}}}}$ ($\\mu$bar)")
    axs[0].set_ylabel(f"A ($\\mu$mol $m^{{-2}}s^{{-1}}$)")
    axs[1].set_xlabel(f"I$_{{\\text{{inc}}}}$ ($\\mu$mol $m^{{-2}}s^{{-1}}$)")
    handles, labels = axs[0].get_legend_handles_labels()
    axs[0].legend(handles, labels, loc='lower right',fontsize=12)
    handles, labels = axs[1].get_legend_handles_labels()
    axs[1].legend(handles, labels, loc='lower right',fontsize=12) 
    fig.savefig(f"datos_obs{N}.png", dpi=300, bbox_inches='tight')
    
    "Datos ajustados con BIC"
    hatA_post = np.median(A_post,axis=0) #Estimadores a posterior de A (por mediana)
    BIC = np.quantile(A_post, [0.025,0.975], axis=0) #BIC
    
    n1 = len(np.array(datos1["C"]))
    n2 = len(np.array(datos2["C"]))
    n3 = len(np.array(datos3["I"]))
    n4 = len(np.array(datos4["I"]))
    

    #Gráficas 
    rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12})
    fig, axs = subplots(1, 2, figsize=(15, 7))

    axs[0].scatter(datos1["C"],datos1["A"], label=f"O$_{{2}}=2\\%$",color="blue")
    axs[0].scatter(datos2["C"],datos2["A"],label=f"O$_{{2}}=21\\%$",color="green")
    axs[1].scatter(datos3["I"],datos3["A"],label=f"O$_{{2}}=2\\%$",color="blue")
    axs[1].scatter(datos4["I"],datos4["A"],label=f"O$_{{2}}=21\\%$",color="green")
    axs[0].set_xlabel(f"C$_{{\\text{{i}}}}$ ($\\mu$bar)")
    axs[0].set_ylabel(f"A ($\\mu$mol $m^{{-2}}s^{{-1}}$)")
    axs[1].set_xlabel(f"I$_{{\\text{{inc}}}}$ ($\\mu$mol $m^{{-2}}s^{{-1}}$)")
    print("Hasta aquí no debe haber problema 1")
    axs[0].plot(datos1["C"],hatA_post[:n1],color="blue")
    axs[0].plot(datos2["C"],hatA_post[n1:n1+n2],color="green")
    axs[1].plot(datos3["I"],hatA_post[n2+n1:n1+n2+n3],color="blue")
    axs[1].plot(datos4["I"],hatA_post[n1+n2+n3:n4+n3+n2+n1],color="green")
    axs[0].fill_between(datos1["C"],BIC[0,:n1],BIC[1,:n1],color="blue",alpha=0.3)
    axs[0].fill_between(datos2["C"],BIC[0,n1:n2+n1],BIC[1,n1:n2+n1],color="green",alpha=0.3)
    axs[1].fill_between(datos3["I"],BIC[0,n2+n1:n2+n1+n3],BIC[1,n2+n1:n2+n1+n3],color="blue",alpha=0.3)
    axs[1].fill_between(datos4["I"],BIC[0,n2+n1+n3:n4+n3+n2+n1],BIC[1,n2+n1+n3:n2+n1+n3+n4],color="green", alpha=0.3)
    handles, labels = axs[0].get_legend_handles_labels()
    axs[0].legend(handles, labels, loc='lower right',fontsize=12)
    handles, labels = axs[1].get_legend_handles_labels()
    axs[1].legend(handles, labels, loc='lower right',fontsize=12)
    fig.savefig(f"datos_ajs{N}.png", dpi=300, bbox_inches='tight')
    print("Se guardó")

