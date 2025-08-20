#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, least_squares
from IPython.display import display, Latex
from fpdf import FPDF 
# Import experimental data methods 
from readdata import *

class pdfReportStyle(FPDF):
    def __init__(self):
        super().__init__()

    def header(self, texto='Header'):
        self.set_font('Arial', '', 12)
        self.cell(0, 8, texto, 0, 1, 'C')    

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', '', 12)
        self.cell(0, 8, f'Page {self.page_no()}', 0, 0, 'C')

class PHSmodel:
    '''
    PHSmodel: Model of photosintesis based in the FvCB model
            for C3 metabolism acording to Yin et. al. (2009)
    '''
    def __init__(self, init_dict):
        '''
        init_dict defines all class variables. 
        
        FORMAT:
        init_dict = {
            'name': [Default value, "Units", "Description"],\
        }
        '''
        for key in init_dict.keys():
            self.__dict__[key] = init_dict[key][0]    
        self.info = init_dict

#### Data Methods ####################

    # Experimental Data upload
    def uploadExpData(self, data_dict):
        '''
        Upload experimental data with a dictionary or dictionaries. 
        
        FORMAT:
        data_dict = {
            'e#'   :{ 
                    'name'      :   'XXXX',
                    '@varconst' :   'C or O',
                    'I'         :   [],
                    'C'         :   [],
                    'O'         :   [],
                    'T'         :   [],
                    'A'         :   [],
                    'Phi2'      :   [],
                    .... other variables may be included... 
                 }
            'e(#+1) :{

            }     

        }
        '''
        self.data = data_dict     
    # Data selection bewteen Ci or O constant experimental values  
    def dataAt(self, constvar='Ci', oxigeno = None, **kwargs):
    
        Qin = np.array([])
        Ci = np.array([])
        O2 = np.array([])
        Tleaf = np.array([])
        A = np.array([])
        PCi = np.array([])
        Error = np.array([])
        
        if constvar == 'Ci':
            Phi2 = np.array([])
            Transmitivity = np.array([])

        # Select experiments according to Oxingen values 
        # none or constant values
        #
        exp_list = []
        if oxigeno == None:
            exp_list.extend(self.data.keys())
        else:                        
           for key in self.data.keys():
                if (self.data[key]['Constant'] == constvar) & (self.data[key]['O2'][0]==oxigeno) :
                    exp_list.append(key)

        # Gather data of the experiments in exp_list
        for key in exp_list:
            if self.data[key]['Constant'] == constvar:
                Qin = np.append(Qin,self.data[key]['Qin'])
                Ci = np.append(Ci,self.data[key]['Ci_Corrected'])
                O2 = np.append(O2,self.data[key]['PO2'])
                Tleaf = np.append(Tleaf,self.data[key]['Tleaf'])
                A = np.append(A,self.data[key]['A_Corrected'])
                PCi = np.append(PCi,self.data[key]['PCi_Corrected'])
                Error = np.append(Error,self.data[key]['Error'])

                if self.data[key]['Constant'] == 'Ci':
                    Transmitivity = np.append(Transmitivity,self.data[key]['Transmssivity'])
                    Phi2 = np.append(Phi2,self.data[key]['Φ2'])
                    
        # Eliminar los primeros M puntos 
        if constvar != 'Ci': 
            M = 5
            Qin = Qin[M:]
            Ci  =  Ci[M:]
            O2  =  O2[M:] 
            Tleaf = Tleaf[M:]
            A = A[M:]
            PCi = PCi[M:]
            Error= Error[M:]

        # elimino valores negativos de Ci, Pci y Qi
        maskCi =  [Ci  > 0]
        maskPCi = [PCi > 0]


        mask = tuple(maskCi and maskPCi)
        Qin     = np.maximum(0,Qin[mask])
        Ci      = Ci[mask]
        O2      = O2[mask]
        Tleaf   = Tleaf[mask]
        A       = A[mask]
        PCi     = PCi[mask]
        Error     = Error[mask]
        
        # Define the dictionary data 
        data_dic = {'I': Qin,'C': Ci,'O': O2,'T': Tleaf,'A': A, 'PCi': PCi, 'Error':Error}   
        
        # Add data for Ci as constant value
        if constvar == 'Ci':
            data_dic['Phi2'] = Phi2[mask]
            data_dic['Transmitivity'] = Transmitivity[mask]

        return data_dic        

    def dataAll(self, **kwargs):

        Qin = np.array([])
        Ci = np.array([])
        O2 = np.array([])
        Osp = np.array([])
        Tleaf = np.array([])
        A = np.array([])
        PCi = np.array([])
        Error = np.array([])


        for key in self.data.keys():
                Qin = np.append(Qin,self.data[key]['Qin'])
                Ci = np.append(Ci,self.data[key]['Ci_Corrected'])
                O2 = np.append(O2,self.data[key]['PO2'])
                Osp = np.append(Osp,self.data[key]['O2'])
                Tleaf = np.append(Tleaf,self.data[key]['Tleaf'])
                A = np.append(A,self.data[key]['A_Corrected'])
                PCi = np.append(PCi,self.data[key]['PCi_Corrected'])
                Error = np.append(Error,self.data[key]['Error'])

        # elimino valores negativos de Ci, Pci y Qi
        maskCi =  [Ci  > 0]
        maskPCi = [PCi > 0]
        

        mask = tuple( maskCi and maskPCi )
        Qin     = np.maximum(0,Qin[mask])
        Ci      = Ci[mask]
        O2      = O2[mask]
        Osp     = Osp[mask]
        Tleaf   = Tleaf[mask]
        A       = A[mask]
        PCi     = PCi[mask]
        Error   = Error[mask]


        data_dic = {
                'I': Qin,
                'C': Ci,
                'O': O2,
                'T': Tleaf,
                'A': A,  
                'Osp' : Osp,
                'PCi' : PCi,
                'Error' : Error 
            }        

        return data_dic        

#### Model functions ####################
          
    def fAjNPR(self, I, s, Rd, **kwargs):
        '''
            Linearization of Aj at  Non Photo Respiratory 
            condition (NPR), namely

            Aj = s * ( I * Phi2/4 ) - Rd  Eq. (7b) 

        '''
        self.__dict__.update(kwargs)
    
        # Load data of experiment at Ci = constants
        dat = self.dataAt('Ci') 
        Phi2 = dat['Phi2'][:len(I)]   
        return s * ( Phi2 * I)/4.0 - Rd  

    def fPhi2(self, I, Theta2, J2max, alfa2_LL, Phi2_LL, **kwargs):
            ''' 
            Quantum efficiency of PSII e- flow Eq. (6)

            Phi2  = (1/2*Theta2*Ro2*Iabs)*(alfa2_LL*Iabs + J2max 
            + [(alfa2_LL*Iabs + J2max)**2-4*Theta2*alfa2_LL*Iabs )]**(1/2)) 
            
            '''
            self.__dict__.update(kwargs)

            Ro2 = alfa2_LL / Phi2_LL
            a = Theta2
            b = - (alfa2_LL * I + J2max)
            c =  alfa2_LL * J2max * I 
            return (-b - np.sqrt(b**2 - 4.0 * a * c))/(2 * a * Ro2 * I)   

    def fJ2(self, I, O, Theta2, J2max, **kwargs):
        ''' 
            Rate of all e- transport through PSII Eq. (4b)

            a = Theta2
            b = -(alfa_LL * Iabs + J2max)
            c= alfa2_LL*J2max*Iabs
            J2 = (b**2-(b**2-4*a*c)**(1/2))/(2*a) 

            self.alfa2_LL is taken from the internal database    
        '''
        self.__dict__.update(kwargs)
        a = Theta2
        b = - (self.alfa2_LL * I + J2max)
        c =  self.alfa2_LL * J2max * I
        return (-b - np.sqrt(b**2 - 4.0 * a * c))/(2.0 * a) 

    def fJ(self, Iinc, Theta, Jmax, k2_LL, **kwargs):
            ''' ec (9) 
                we use this equation to estimate Jmax and Theta
            '''
            self.__dict__.update(kwargs)

            a = Theta
            b = - (k2_LL*Iinc + Jmax)
            c =  k2_LL*Jmax*Iinc
            result =  (-b - np.sqrt(b**2-4.0*a*c))/(2*a)
        
            return result 

    def fGamma_str(self, O, Sco,  **kwargs):
        '''
            Cc compensation points in the absence of Rd 
            (Definition after Eq.(2))

            Gamma_str = 0.5 * O / Sco
        '''
        self.__dict__.update(kwargs)

        return 0.5 * O / Sco 
        
    def fAp(self, Tp, Rd):
        '''
            Triose phosphate utilization limited 
            net photosynthesis rate Eq. (5)
            Ap = 3.0 * Tp - Rd
        '''
        return 3.0*Tp - Rd
 
    def fAc(self, I, Ci, O, T, gmo, delta, Sco, Kmc, Kmo, Vcmax, Rd):
        '''
            Rubisco activity limitation rate Eq. (12)
            Ac = 

            Some units consistency changes are needed:
                [Gammastr] =  Pa, requerido [Gammastr] = microbar (microbar = 10.0 * Pa)
                [Kmo] = kPa, requerido  [Kmo] = milibar (milibar = 10*kP)
                [Kmc] = Pa, requerido   [Kmc] = microbar (microbar = 10*P)
        '''

        # Variables needed as vectors of len(T)
        Gammastr = self.fGamma_str(O, Sco)*np.ones(len(T))
        Kmo = 10.0*Kmo*np.ones(len(T))
        Kmc = 10.0*Kmc*np.ones(len(T))

        # Aux. variables Ec. (12) 
        x1 = Vcmax * np.ones(I.size)
        x2 = Kmc * (1.0 + O/Kmo) 

        # Parameters for cuadratic equation  Eq. (12) for Ac
        a =  x2 + Gammastr + delta * ( Ci + x2)   
        b = -((x2 +  Gammastr) * (x1 - Rd)+ (Ci + x2) * (gmo * (x2 +  Gammastr) + delta * (x1 - Rd)) + delta * (x1 * (Ci - Gammastr) - Rd * (Ci + x2)))
        c = (gmo * (x2 + Gammastr) + delta * (x1 - Rd))*(x1 * (Ci - Gammastr) - Rd * (Ci + x2))    
   
        return (-b - np.sqrt(b**2 - 4.0 * a * c))/(2.0 * a)     
    
    def fAj(self, I, Ci, O, T, gmo, delta, Sco, Kmc, Kmo, theta, Jmax, k2_LL, Rd):
        '''
            Rubisco activity limitation rate Eq. (12)
            Aj = 

            Some units consistency changes are needed:
                [Gammastr] =  Pa, requerido [Gammastr] = microbar (microbar = 10.0 * Pa)
                [Kmo] = kPa, requerido  [Kmo] = milibar (milibar = 10*kP)
                [Kmc] = Pa, requerido   [Kmc] = microbar (microbar = 10*P)
        '''
        # Variables needed as vectors of len(T)
        Gammastr = self.fGamma_str(O, Sco)*np.ones(len(T))
        Kmo = 10.0*Kmo*np.ones(len(T))
        Kmc = 10.0*Kmc*np.ones(len(T))

        # Aux. variables Ec. (12) 
        x1 = self.fJ(I, theta, Jmax, k2_LL)/4.0            
        x2 = 2.0 * Gammastr

        # Parameters for cuadratic equation  Eq. (12) for Aj
        a =  x2 + Gammastr + delta * ( Ci + x2)   
        b = -((x2 +  Gammastr) * (x1 - Rd)+ (Ci + x2) * (gmo * (x2 +  Gammastr) + delta * (x1 - Rd)) + delta * (x1 * (Ci - Gammastr) - Rd * (Ci + x2)))
        c = (gmo * (x2 + Gammastr) + delta * (x1 - Rd))*(x1 * (Ci - Gammastr) - Rd * (Ci + x2))    
 
        return (-b - np.sqrt(b**2 - 4.0 * a * c))/(2.0 * a)  

    def fAsimple(self, P, gmo, Vcmax, delta, Tp, theta, Jmax, k2_LL, Sco, Kmo, Kmc, Rd):
        '''
            Combination of all limitation rates Eq. (1) 
            A = min(Ap, Aj, Ac)
            P - experimental data 
        ''' 
        I, C, O, T = P 

        Ap = self.fAp(Tp*np.ones(I.shape), Rd)
        Ac = self.fAc(I, C, O, T, gmo, delta, Sco, Kmc, Kmo, Vcmax, Rd)
        Aj = self.fAj(I, C, O, T, gmo, delta, Sco, Kmc, Kmo, theta, Jmax, k2_LL, Rd)
        
        return np.minimum( np.minimum(Ap, Ac), Aj)

    def fABayes(self, P, gmo, Vcmax, delta, Tp, thetaNPR, thetaPR, JmaxNPR, JmaxPR, k2_LLNPR, k2_LLPR, Sco, Kmo, Kmc, Rd_NPR, Rd_PR):
        '''
            Combination of all limitation rates Eq. (1) 
            A = min(Ap, Aj, Ac)
            P - experimental data 
        ''' 
        #I, C, O, T, Osp = P 
        NPR_Index = P[4,:]== 20

        I, C, O, T = P[0:4,NPR_Index] # Datos NPR
        Ap_NPR = self.fAp(Tp*np.ones(I.shape), Rd_NPR)
        Ac_NPR = self.fAc(I, C, O, T, gmo, delta, Sco, Kmc, Kmo, Vcmax, Rd_NPR)
        Aj_NPR = self.fAj(I, C, O, T, gmo, delta, Sco, Kmc, Kmo, thetaNPR, JmaxNPR, k2_LLNPR, Rd_NPR)
        
        I, C, O, T  = P[0:4,~NPR_Index] # datos PR
        Ap_PR = self.fAp(Tp*np.ones(I.shape), Rd_PR)
        Ac_PR = self.fAc(I, C, O, T, gmo, delta, Sco, Kmc, Kmo, Vcmax, Rd_PR)
        Aj_PR = self.fAj(I, C, O, T, gmo, delta, Sco, Kmc, Kmo, thetaPR, JmaxPR, k2_LLPR, Rd_PR)    
        
        # Juntamos las evaluaciones NPR y PR
        # Arreglos para juntar evaluaciones
        Ap = np.zeros(len(NPR_Index))
        Ac = np.zeros(len(NPR_Index))
        Aj = np.zeros(len(NPR_Index))
        #Ap
        Ap[NPR_Index] = Ap_NPR
        Ap[~NPR_Index] = Ap_PR
        #Ac
        Ac[NPR_Index] = Ac_NPR
        Ac[~NPR_Index] = Ac_PR
        #Aj
        Aj[NPR_Index] = Aj_NPR
        Aj[~NPR_Index] = Aj_PR

        return np.minimum( np.minimum(Ap, Ac), Aj)

    def fA(self, P, gmo, Vcmax, delta, Tp, thetaNPR, thetaPR, JmaxNPR, JmaxPR, k2_LLNPR, k2_LLPR, Sco, Kmo, Kmc):
        '''
            Combination of all limitation rates Eq. (1) 
            A = min(Ap, Aj, Ac)
            P - experimental data 
        ''' 
        #I, C, O, T, Osp = P 
        NPR_Index = P[4,:]== 20

        I, C, O, T = P[0:4,NPR_Index] # Datos NPR
        Ap_NPR = self.fAp(Tp*np.ones(I.shape), self.Rd_NPR)
        Ac_NPR = self.fAc(I, C, O, T, gmo, delta, Sco, Kmc, Kmo, Vcmax, self.Rd_NPR)
        Aj_NPR = self.fAj(I, C, O, T, gmo, delta, Sco, Kmc, Kmo, thetaNPR, JmaxNPR, k2_LLNPR, self.Rd_NPR)
        
        I, C, O, T  = P[0:4,~NPR_Index] # datos PR
        Ap_PR = self.fAp(Tp*np.ones(I.shape), self.Rd_PR)
        Ac_PR = self.fAc(I, C, O, T, gmo, delta, Sco, Kmc, Kmo, Vcmax, self.Rd_PR)
        Aj_PR = self.fAj(I, C, O, T, gmo, delta, Sco, Kmc, Kmo, thetaPR, JmaxPR, k2_LLPR, self.Rd_PR)    
        
        # Juntamos las evaluaciones NPR y PR
        # Arreglos para juntar evaluaciones
        Ap = np.zeros(len(NPR_Index))
        Ac = np.zeros(len(NPR_Index))
        Aj = np.zeros(len(NPR_Index))
        #Ap
        Ap[NPR_Index] = Ap_NPR
        Ap[~NPR_Index] = Ap_PR
        #Ac
        Ac[NPR_Index] = Ac_NPR
        Ac[~NPR_Index] = Ac_PR
        #Aj
        Aj[NPR_Index] = Aj_NPR
        Aj[~NPR_Index] = Aj_PR

        return np.minimum( np.minimum(Ap, Ac), Aj)

    def fASco(self, P, gmo, Vcmax, delta, thetaNPR, thetaPR, JmaxNPR, JmaxPR, k2_LLNPR, k2_LLPR, Kmo, Kmc):
        '''
            Here we use the estimate of Sco self.estimateSco()
            and the estimae of self.estimateTp() 
        ''' 
        Sco = self.Sco
        return self.fA( P, gmo, Vcmax, delta, self.Tp, thetaNPR, thetaPR, JmaxNPR, JmaxPR, k2_LLNPR, k2_LLPR, Sco, Kmo, Kmc)

#### Estimation methods  ################

    def estimateRdSc(self, oxigeno = [20, 210], **kwargs):
        ''' 
            Estimate of Rd and s by fAjNRP
            Data is re-scaled version of experiment 
            at oxigeno = 20 
        '''        
        for ox in oxigeno:
            #load rescaled data at oxigeno = 20    
            dat = self.interpolatePhi2atAIvalues(oxigeno = ox)
            # Fitting
            popt, pcov = curve_fit(self.fAjNPR, 
                                xdata = dat['I'], 
                                ydata = dat['A'], 
                                bounds=([0,0], [np.inf,np.inf]), # Se pueden poner cotas a los valores de s y Rd
                                **kwargs)
            # Actualizo los diccionarios
            if ox == 20:
                self.s, self.Rd_NPR = popt
            elif ox == 210:
                s_, self.Rd_PR = popt

        return True

    def estimateTp(self, **kwargs):
        # estimacion de Tp en Ap = 3*Tp-Rd
        dat = self.dataAll()
        self.Tp = (1.05*max(dat['A'])+max(self.Rd_PR,self.Rd_NPR))/3.0


    def estimateJmax2Theta2(self, datatype='measured', **kwargs):
            ''' 
            Estimates of [Theta2, J2max] using equation fPhi2
            
            Data:
                Beta, Iinc, Phi2

            Flags:
                datatype = 'measured' => Phi2 from the data base
                datatype = 'estimated' => Phi2 estimated from Jmax, Theta,    
            '''
            # Data load 
            if datatype == 'measured':
                dat = self.dataAt('Ci')
            elif datatype == 'estimated':
                dat = self.estimatePhi2()  
  
            Iabs = self.Beta * dat['I']
            # Fitting
            popt, pcov = curve_fit(self.fPhi2, 
                                    xdata = Iabs, 
                                    ydata = dat['Phi2'], 
                                    #       Theta2, J2max, alfa2_LL, Phi2_LL,,
                                    bounds=([0, 0.1, 0, 0], [0.7, 400, 1, 1]), # Se pueden poner cotas a los valores de s y Rd
                                    **kwargs)
            
            # Actualizo los diccionarios
            if datatype == 'measured':
                self.Theta2, self.J2max, self.alfa2_LL, self.Phi2_LL = popt
            elif datatype == 'estimated':
                self.Theta2est, self.J2maxest, self.alfa2_LLest, self.Phi2_LLest = popt      

            return

    def estimateJmaxTheta(self, datatype = 'measured', **kwargs):
        #Calculating e- transport parameters k2(LL), Jmax and q
        '''
        Estimates of self.Theta, self.Jmax, self.k2_LL
        Inputs:
                Beta, Iinc, Phi2
        Flags:
                Phi2 = 'measured' => Phi2 from the data base
                Phi2 ='estimated' => Phi2 estimated from Jmax, Theta,    
            '''

        # Data load 
        if datatype == 'measured':
            dat = self.dataAt('Ci', oxigeno= 210)
        elif datatype == 'estimated':
            dat = self.estimatePhi2( oxigeno = 20)  
  
        self.k2     = self.s*dat['Phi2']
        self.k2_LL  = self.s*self.Phi2_LL
        self.J      = self.s*dat['I'] * dat['Phi2']

        # Fitting
        popt, pcov = curve_fit(self.fJ, 
                                xdata = dat['I'], 
                                ydata = self.J, 
                                bounds=([0,0.1,0.0], [2.0,400,5.0]), # Se pueden poner cotas a los valores de s y Rd
                                **kwargs)
            
        # Actualizo los diccionarios    
        if datatype == 'measured':
            self.Theta, self.Jmax, self.k2_LL = popt
        elif datatype == 'estimated':
            self.Thetaest, self.Jmaxest, self.k2_LLest = popt    

        return True

    def estimateSco(self, N = 5):
        
        datLow  =   self.dataAt(constvar='Qin', oxigeno = 20)
        datHigh =   self.dataAt(constvar='Qin', oxigeno = 210)

        # fit blow that is the slope in the Alow vs Clow relationship 
        ALow =  datLow['A'][-N:]
        CLow =  datLow['C'][-N:]
        OLow =  datLow['O'][-N:]
        popt, pcov = curve_fit(self.line, 
                                xdata = CLow, 
                                ydata = ALow,
                            #VarsOrder =  m,    b
                                bounds=([0.0,  -10.0], 
                                        [5.0,   10.0])
                                )         
        self.bLow, b_ = popt

        # fit blow that is the slope in the Alow vs Clow relationship 
        AHigh =  datHigh['A'][-N:]
        CHigh =  datHigh['C'][-N:]
        OHigh =  datHigh['O'][-N:]

        popt, pcov = curve_fit(self.line, 
                                xdata = CHigh, 
                                ydata = AHigh, 
     
                            #VarsOrder =  m,    b
                                bounds=([0.0,  -10.0], 
                                        [5.0,   10.0])
                                )   
        self.bHigh, b_ = popt

        ydat = (2/(OHigh-OLow))*((ALow+self.Rd_NPR)/self.bLow -(AHigh+self.Rd_PR)/self.bHigh)
        xdat = (2/(OHigh-OLow))*(CHigh-CLow)
        #plt.plot(xdat,ydat,'.')
        #plt.show()
        popt, pcov = curve_fit(self.line0, 
                                xdata = xdat, 
                                ydata = ydat, 
                                #VarsOrder =  1/Sc/o, 
                                bounds=(0.0,10.0)
                                )   
        self.Sco = 1/popt


        self.Gamma_strNPR = self.fGamma_str(datLow['O'][0], self.Sco)
        self.Gamma_strPR = self.fGamma_str(datHigh['O'][0], self.Sco)
        print('El valor de Sco =' + str(self.Sco))
        return True
        
    def estimateFvCBmain(self, **kwargs ):
        # Ajustes para:  gmo, Vcmax, delta, Tp, 
        #                thetaNPR, thetaPR, JmaxNPR, JmaxPR, k2_LLNPR, k2_LLPR, 
        #                Kmo, Kmc

        dat = self.dataAll()
        # Set-up experimental data
        self.P = np.array([dat['I'], dat['PCi'], dat['O'], dat['T'], dat['Osp']]) 

        popt, pcov = curve_fit(self.fASco, 
                                xdata = (dat['I'], dat['PCi'], dat['O'], dat['T'], dat['Osp']), 
                                ydata = dat['A'], 
                            # gmo, Vcmax, delta, Tp, thetaNPR, thetaPR, JmaxNPR, JmaxPR, k2_LLNPR, k2_LLPR, Sco, Kmo, Kmc
                            #VarsOrder =   gmo, Vcmax, delta, ThetaNPR, ThetaPR, JmaxNPR, JmaxPR, k2_LLNPR, k2_LL_PR,  Kmo,  Kmc
                                bounds=([  0.0,  10.0,   0.0,       0.0,     0.0,     2.0,    2.0,      0.1,      0.1,  1.0,  1.0 ], 
                                        [0.01, 200.0,  40.0,      1.0,     1.0,   500.0,  500.0,      0.9,      0.9, 50.0, 50.0 ]), # Se pueden poner cotas a los valores de s y Rd
                                        **kwargs)

        # Dictionary update
        self.gmo, self.Vcmax, self.delta, self.Theta_NPR, self.Theta_PR, self.Jmax_NPR, self.Jmax_PR, self.k2_LL_NPR, self.k2_LL_PR, self.K_mO, self.K_mC = popt 
        
        return True

    def estimate(self):
        # las estimaciones se hace dobles, con NPR y no PR conditions
        self.estimateRdSc()
        self.estimateTp()
        self.estimateSco()
        self.estimateFvCBmain()

        self.estimateJmax2Theta2(datatype = 'estimated')
        self.estimateJmaxTheta(datatype = 'estimated')
        return True

#### Reconstruction Phi2 methods #######################

    def line(self,x,m,b):
        '''
            auxiliary linear function
        '''
        return m * x + b

    def line0(self, x,b):
        return self.line(x,-1,b)

    def interpolatePhi2atAIvalues(self, oxigeno = None):
        ''' 
            Oxigeno = None, 20, 210
            Method to ESTIMATE Phi2 at LOW I values 
            and it corresponding A values. 
        '''
        # Data to adjust
        dat  = self.dataAt('Ci', oxigeno = oxigeno)   
        I    = dat['I']
        Ib   = (dat['Transmitivity']) * I
        Phi2 = dat['Phi2']
        A    = dat['A']


        # Fit of linear relationship between Phi2 = m * Ib +b
        popt, pcov = curve_fit(self.line, 
                                xdata  = Ib, 
                                ydata  = Phi2, 
                                bounds = ([-100,0], [np.inf,np.inf]))
        
        # Select the values of I and A to use.
        Icut = I[ I < 2*max(Ib) ]  # Revisar porque funciona?<.----
        Acut = A[ I < 2*max(Ib) ]
        # Estimate Phi2 at Icut values
        m, b = popt
        Phi3cut = self.line(Icut, m , b)
        # Gather  Acut, Icut, Phi2cut values
        df = {
            'A'     : Acut,
            'I'     : Icut,
            'Phi2'  : Phi3cut
        }
        return df    

    def estimatePhi2(self, oxigeno = 20, Ilow = 800, Ilarge = 850):
        '''
            Estimate of Phi2 values for [low,large] I values
            I[low] < Ilow & I[large]> Ilarge
        '''
        # Estimate of Phi2 values for low I
        interpolated_data = self.interpolatePhi2atAIvalues(oxigeno = oxigeno)
        I1 = interpolated_data['I']
        Phi2_1 = interpolated_data['Phi2']
        A1 = interpolated_data['A']

        # Estimate of Phi2 values for large I
        dat = self.dataAt('Ci', oxigeno = oxigeno) 
        I2 = np.array(dat['I'])
        A2 = np.array(dat['A'])
        J_2 = self.fJ(I2, self.Theta_NPR, self.Jmax_NPR, self.k2_LL_NPR)
        #  Phi_2 estimated by J = s * Phi2 * I, namely
        Phi2_2 = J_2/(I2 * self.s)

        # Gather data for I < Ilow and Ilarge < I values
        dat={}
        dat['A']    = np.append(A1[ I1 < Ilow ], A2[ I2 > Ilarge ])
        dat['Phi2'] = np.append(Phi2_1[ I1 < Ilow ], Phi2_2[I2 > Ilarge ])
        dat['I']    = np.append(I1[ I1 < Ilow ], I2[ I2 > Ilarge ])
        return dat

#### Plot and reporting methods #######################

#### Plots         
    def plot_experiment(self, oxigeno, constParam = 'Ci', gmo = None, Vcmax = None, delta = None, Tp = None):
        '''
            Plot experiments at C or Qin constant
        '''

        if gmo == None :
            gmo = self.gmo
        if Vcmax == None:
            Vcmax = self.Vcmax
        if delta == None :
            delta = self.delta
        if Tp == None :
            Tp = self.Tp

        # Filtro los datos al experimento con 'C' constante
        dat = self.dataAt(constParam, oxigeno = oxigeno)
        if oxigeno == 20: 
            pcolor = 'darkorange'
            # Model evaluation      
            P = np.array([dat['I'], dat['PCi'], dat['O'], dat['T']]) 
            Y = self.fAsimple(P, gmo, Vcmax, delta, Tp, self.Theta_NPR, self.Jmax_NPR, self.k2_LL_NPR, self.Sco, self.K_mO, self.K_mC, self.Rd_NPR)
                
        else:
            pcolor = 'steelblue'
            P = np.array([dat['I'], dat['PCi'], dat['O'], dat['T']]) 
            Y = self.fAsimple(P, gmo, Vcmax, delta, Tp, self.Theta_PR, self.Jmax_PR, self.k2_LL_PR, self.Sco, self.K_mO, self.K_mC, self.Rd_PR)

        if constParam == 'Ci':
            # Plot model
            plt.plot(dat['I'], Y, color=pcolor, linestyle='dashed',
                    lw = 2#, label = '$O_2$ = %3.0f' % oxigeno
                    )   
        
            # Plot data
            plt.scatter(dat['I'],dat['A'], color = pcolor, 
                    alpha=1.0, label= '$O_2$ = %3.0f' % oxigeno,s=100) 
            plt.xlabel('$I_{inc}$ [$\\mu$mol m$^{-2}$ s$^{-1}$]', fontsize=42)
            plt.ylabel('A  [$\\mu$mol m$^{-2}$ s$^{-1}$]', fontsize=42)
            
            X_labels = [0,500, 1500,2000]
            Y_labels= [-5,0,5,15,25]

            plt.xticks(X_labels, fontsize = 41)
            plt.yticks(Y_labels, fontsize = 41)
            #plt.title('Experiment @Ci=const')
        elif constParam == 'Qin':
            # Plot model
            plt.plot(dat['C'], Y, color = pcolor, linestyle='dashed', 
                    lw=2#,label='fit: E2'
                    ) 
             
            # Plot data 
            X_labels = [0,400, 800,1200]
            Y_labels= [0,10,20,30]
            
            plt.scatter(dat['C'], dat['A'], color = pcolor,
                    alpha = 0.9,label = '$O_2$ = %3.0f' % oxigeno, s=100) 
            plt.xlabel('$C_{i}$ [ppm]',fontsize=42)
            plt.ylabel('A  [$\\mu$mol m$^{-2}$ s$^{-1}$]',fontsize=42)
            
            plt.xticks(X_labels,fontsize = 41)
            plt.yticks(Y_labels, fontsize = 41)
            #plt.title('Experiment @Qin=const')

        plt.legend(fontsize=40, loc="lower right")   
        plt.rcParams['figure.figsize'] = [15, 13]
        #figsize
        #plt.figure(figsize=(20,18))
        
        return True

    def plotPhi2(self,  estimate = False ):
        '''
            Plot of I vs Phi2 for meassured or estimated values
        '''
        # Sample's name
        name = self.data['e2']['Sample_ID'][0]  
        if estimate == True:
            Theta2   = self.Theta2est
            J2max    = self.J2maxest
            alfa2_LL = self.alfa2_LLest
            Phi2_LL  = self.Phi2_LLest
            dat      = self.estimatePhi2()
        elif estimate == False:
            Theta2   = self.Theta2est
            J2max    = self.J2maxest
            alfa2_LL = self.alfa2_LLest
            Phi2_LL  = self.Phi2_LLest
            dat      = self.dataAt('Ci', oxigeno = 20)      

        # I values for the model evaluation
        I = np.linspace( min( dat['I'] ), max( dat['I'] ), 20)
        # Plot model simulation
        plt.plot(I, self.fPhi2(self.Beta * I, Theta2, J2max, alfa2_LL, Phi2_LL), 
        color = 'lightslategray', lw = 2,label = 'fit: $\\theta_2$ = %5.3f, $J2_{max}$ = %5.3f' 
        % tuple([Theta2, J2max ] ))         
        # Plot data points
        plt.scatter(dat['I'], dat['Phi2'], color = 'darkslategray',
                    label = '$\\Phi_2$ vs $I_{inc}$ ')         
        plt.xlabel('$I_{inc}$ [' + self.info['Iinc'][1] + ']')
        plt.ylabel('$\\Phi_2$ [' + self.info['Phi2'][1] + ']')
        plt.legend()
        plt.title(name )
        return True

    def plotJ(self ):
        '''
            Plot I vs estimate of J
        '''
        # Sample's name
        name  = self.data['e1']['Sample_ID'][0]
        dat = self.dataAt('Ci', oxigeno = 20)
        
        I = dat['I']
        y = self.fJ(I, self.Theta_NPR, self.Jmax_NPR, self.k2_LL_NPR)
        plt.plot(I, y, color = 'darkorange', lw = 2,
                label = 'fit: $\\theta$ = %5.3f, $J_{max}$ = %5.3f'
                % tuple([self.Theta_NPR, self.Jmax_NPR ]))
        
        y1 = self.fJ(I, self.Theta_PR, self.Jmax_PR, self.k2_LL_PR)
        plt.plot(I, y1, color = 'steelblue', lw = 2,
                label = 'PR, fit: $\\theta$ = %5.3f, $J_{max}$ = %5.3f'
                % tuple([self.Theta_PR, self.Jmax_PR ]))
        
        plt.xlabel('$I_{inc}$ [' + self.info['Iinc'][1] + ']')
        plt.ylabel('J  [' + self.info['J'][1] + ']')
        plt.legend()
        plt.title(name)  

    def plotReport(self, save=True):
        
        # Plots experiments @Ci = const
        self.plot_experiment(oxigeno =  20, constParam = 'Ci')
        self.plot_experiment(oxigeno = 210, constParam = 'Ci')
        if save == True:
            plt.savefig('C:/Users/nes_3/OneDrive/Escritorio/Fotosintesis/IvsA.pdf', 
            transparent=False,  
            facecolor='white', 
            bbox_inches="tight")
            plt.show()
        else:    
            plt.show()

        # Plots experiments @Qin = const
        self.plot_experiment(oxigeno =  20, constParam = 'Qin')
        self.plot_experiment(oxigeno = 210, constParam = 'Qin')
        if save == True:
            plt.savefig('C:/Users/nes_3/OneDrive/Escritorio/Fotosintesis/CvsA.pdf', 
            transparent = False,  
            facecolor   = 'white', 
            bbox_inches = "tight")
            plt.show()
        else:    
            plt.show()

        # Plots of Phi2 estimates
        self.plotPhi2(estimate = True)
        if save == True:
            plt.savefig('C:/Users/nes_3/OneDrive/Escritorio/Fotosintesis/IvsPhi2.png', 
            transparent = False,  
            facecolor   = 'white', 
            bbox_inches = "tight")
            plt.show()
        else:    
            plt.show()    

        # Plot J estimates
        self.plotJ()
        if save == True:
            plt.savefig('C:/Users/nes_3/OneDrive/Escritorio/Fotosintesis/IvsJ.png', 
            transparent = False,  
            facecolor   = 'white', 
            bbox_inches = "tight")
            plt.show()
        else:    
            plt.show()      

    def paramReport(self, data = 'estimated'):
        '''
            Report of the estimated parameters
        '''
        if data == 'measured':     
            dfajustes= pd.DataFrame(
                {'Parametro': ['Jmax','Theta','Rd','s', 
                            'gmo','Vcmx','delta', 'Tp','k2_LL',
                            'Theta2','J2max','alfa2LL','Phi2LL',
                            'Gamma_str', 'Sco', 'K_mO', 'K_mC'],
                'Valor_NPR': [ '%5.2f'% self.Jmax_NPR,
                           '%5.2e'% self.Theta_NPR,
                            '%5.2f'% self.Rd_NPR,
                            '%5.2f'% self.s,
                            '%5.2e'% self.gmo,
                            '%5.2f'% self.Vcmax,
                            '%5.2f'% self.delta,
                            '%5.2f'% self.Tp,
                            '%5.2f'% self.k2_LL_NPR,
                            '%5.2f'% self.Theta2est, 
                            '%5.2f'% self.J2maxest, 
                            '%5.2f'% self.alfa2_LLest, 
                            '%5.2f'% self.Phi2_LLest,
                            '%5.2f'% self.Gamma_strNPR,
                            '%5.2f'% self.Sco,
                            '%5.2f'% self.K_mO, 
                            '%5.2f'% self.K_mC,  
                            ],
                'Valor_PR': [ '%5.2f'% self.Jmax_PR,
                           '%5.2e'% self.Theta_PR,
                            '%5.2f'% self.Rd_PR,
                            '-',
                            '-',
                            '-',
                            '-',
                            '-',
                            '%5.2f'% self.k2_LL_PR,
                            '%5.2f'% self.Theta2est, 
                            '%5.2f'% self.J2maxest, 
                            '%5.2f'% self.alfa2_LLest, 
                            '%5.2f'% self.Phi2_LLest,
                            '%5.2f'% self.Gamma_strPR,
                            '-',
                            '-', 
                            '-',  
                            ],
                'Unidades':[ self.info['Jmax'][1], 
                                self.info['Theta'][1],
                                self.info['Rd'][1],
                                self.info['s'][1],
                                self.info['gmo'][1],
                                self.info['Vcmax'][1],
                                self.info['delta'][1],
                                self.info['Tp'][1],
                                self.info['k2_LL'][1],
                                self.info['Theta2'][1], 
                                self.info['J2max'][1], 
                                self.info['alfa2_LL'][1], 
                                self.info['Phi2_LL'][1],
                                self.info['Gamma_star'][1],
                                self.info['Sco'][1],
                                self.info['K_mO'][1],
                                self.info['K_mC'][1] 
                                ]}
                            )  
        if data == 'estimated':
            dfajustes= pd.DataFrame(
                    {'Parametro': ['Jmax','Theta','Rd','s', 
                                'gmo','Vcmx','delta', 'Tp','k2_LL',
                                'Theta2','J2max','alfa2LL','Phi2LL','Gamma_str','Sco'],
                               'Valor_NPR': [ '%5.2f'% self.Jmax_NPR,
                           '%5.2e'% self.Theta_NPR,
                            '%5.2f'% self.Rd_NPR,
                            '%5.2f'% self.s,
                            '%5.2e'% self.gmo,
                            '%5.2f'% self.Vcmax,
                            '%5.2f'% self.delta,
                            '%5.2f'% self.Tp,
                            '%5.2f'% self.k2_LL_NPR,
                            '%5.2f'% self.Theta2_NPR, 
                            '%5.2f'% self.J2max_NPR, 
                            '%5.2f'% self.alfa2_LL_NPR, 
                            '%5.2f'% self.Phi2_LL_NPR,
                            '%5.2f'% self.Gamma_strNPR,
                            '%5.2f'% self.Sco,
                            ],
                'Valor_PR': [ '%5.2f'% self.Jmax_PR,
                           '%5.2e'% self.Theta_PR,
                            '%5.2f'% self.Rd_PR,
                            '-',
                            '-',
                            '-',
                            '-',
                            '-',
                            '%5.2f'% self.k2_LL_PR,
                            '%5.2f'% self.Theta2_PR, 
                            '%5.2f'% self.J2max_PR, 
                            '%5.2f'% self.alfa2_PR, 
                            '%5.2f'% self.Phi2_LL_PR,
                            '%5.2f'% self.Gamma_strPR,
                            '-',
                            '-', 
                            '-',  
                            ],            
                    'Unidades':[self.info['Jmax'][1], 
                                self.info['Theta'][1],
                                self.info['Rd'][1],
                                self.info['s'][1],
                                self.info['gmo'][1],
                                self.info['Vcmax'][1],
                                self.info['delta'][1],
                                self.info['Tp'][1],
                                self.info['k2_LL'][1],
                                self.info['Theta2'][1], 
                                self.info['J2max'][1], 
                                self.info['alfa2_LL'][1], 
                                self.info['Phi2_LL'][1], 
                                 self.info['Gamma_star'][1],
                                self.info['Sco'][1]
                                ]}
                            )  
        return dfajustes

    def pdfReport(self, datatype = 'measured'):
        #CreatePage
        ch = 8
        pdf = pdfReportStyle()
        pdf.add_page()
        pdf.header('Regresión experimentos sin Phi2')
        pdf.set_font('Arial', '', 12)
        pdf.cell(w = 30, h = ch, txt = "Muestra: ", ln = 0)
        pdf.cell(w = 30, h = ch, txt = self.data['e2']['Sample_ID'][0], ln = 1)
        pdf.ln(ch)
        # Add plots
        pdf.image('C:/Users/nes_3/OneDrive/Escritorio/Fotosintesis/CvsA.pdf', 
                x = 10, y = 35, w = 95, h = 0, type = 'PNG')
        pdf.image('C:/Users/nes_3/OneDrive/Escritorio/Fotosintesis/IvsA.pdf', 
                x = 105, y = 35, w = 95, h = 0, type = 'PNG') 
        # Spacing
        pdf.multi_cell(w=0, h=145, txt=" ")
        pdf.ln(ch)
        # Add light experiments
        pdf.image('C:/Users/nes_3/OneDrive/Escritorio/Fotosintesis/IvsJ.png', 
                  x = 10, y = 115, w = 95, h = 0, type = 'PNG')
        pdf.image('C:/Users/nes_3/OneDrive/Escritorio/Fotosintesis/IvsPhi2.png', 
                  x = 105, y = 115, w = 95, h = 0, type = 'PNG')             
        #pdf.ln(ch)

        # Add data table of adjusted parameters
        df = self.paramReport(data  = datatype)
        pdf.set_font('Arial', '', 10)
        page_width = pdf.w - 2 * pdf.l_margin
        col_width = page_width/4
        pdf.ln(1)
        th = 1.3*pdf.font_size
        # Table Header
        for column in df.columns:
            pdf.cell(col_width, th, column, border=1, align = 'C')
        pdf.ln(th)

        for i in range(0, len(df)):
            for column in df.columns:
                pdf.cell(col_width, th, txt=df[column].iloc[i], border=1, ln = 0, align='C')
            pdf.ln(th)                             
      
        nombrearch = str(self.data['e2']['Sample_ID'][0]+'.pdf')
        pdf.output(f'C:/Users/nes_3/OneDrive/Escritorio/Fotosintesis/IvsJ'+nombrearch, 'F')
        return True

if __name__=="__main__":
    
    # Importa dictionary
    from DataDict import init_dict



    for jk in [x for x in range(31,32) if (x != 17) ]:
        
        # Model instance
        modexp = PHSmodel(init_dict)
        
        # Upload weath data
        datos_trigo = datosendic(oxigeno = [20, 210], N = jk)
        modexp.uploadExpData(datos_trigo)

        # Model regresion
        modexp.estimate()

        modexp.plotReport()
        modexp.pdfReport(datatype = 'measured')
        

    print('Acabe\n')

