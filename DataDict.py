'''
Definimos los diccionario para el ajuste
'''
# Diccionario da variables
init_dict = { \
'A' :           [None, "mu_mol CO2 m**-2 s**-1", "Net photosynthesis rate", "variable"],\
'Ac' :          [None, "mu_mol CO2 m**-2 s**-1", "Rubisco activity limited net photosynthesis rate", "variable"],\
#'Cc' :         [None, "mu_bar", "Chloroplast CO2 partial pressure", "parameter"],\
'Gamma_star25': [3.743, "Pa", "Cc-based CO2 compensation point in the absence of Rd at 25C","parameter"],\
'Vcmax':        [58.5, "mu_mol CO2 m**-2 s**-1", "Maximum rate of Rubisco activity-limited carboxylation","parameter"],\
'Vcmax25':      [58.5, "mu_mol CO2 m**-2 s**-1", "Maximum rate of Rubisco activity-limited carboxylation","parameter"],\
'K_mC25':       [27.238, "Pa", "Michaelis–Menten constant of Rubisco for CO2 at 25C","parameter"],\
'K_mO25':       [16.582, "kPa", "Michaelis–Menten constant of Rubisco for O2 at 25C","parameter"],\
'K_mC':         [27.238, "Pa", "Michaelis–Menten constant of Rubisco for CO2", "variable"],\
'K_mO':         [16.582, "kPa", "Michaelis–Menten constant of Rubisco for O2 ", "variable"],\
'Gamma_star':   [3.743, "mu_bar", "Cc-based CO2 compensation point in the absence of Rd ","variable"],\
'Sco':          [2.93, "mbar mu_bar**-1","Relative CO2/O2 specificity factor for Rubisco", "parameter"],\
'Rd':           [-2.0, "mu_mol CO2 m**-2 s**-1", "Day respiration (i.e. respiratory CO2 release other than by photorespiration)", "parameter"],\
'AH':           [None, "mu_mol CO2 m**-2 s**-1", "Net photosynthesis rate under high O2 condition", "variable"],\
'Aj':           [None, "mu_mol CO2 m**-2 s**-1", "Electron transport limited net photosynthesis rate", "variable"],\
'AL':           [None, "mu_mol CO2 m**-2 s**-1", "Net photosynthesis rate under low O2 condition", "variable"],\
'Ap':           [None, "mu_mol CO2 m**-2 s**-1", "Triose phosphate utilization limited net photosynthesis rate", "variable"],\
'bH':           [None, "mu_bar", "Slope of the initial linear part of the A–Ci curve under a high O2 condition", "parameter"],\
'bL':           [None, "mu_bar", "Slope of the initial linear part of the A–Ci curve under low O2 condition", "parameter"],\
'Ca':           [None, "mu_bar or mu_mol mol-1", "Ambient air CO2 partial pressure or concentration", "variable"],\
'Ci':           [None, "mu_bar", "Intercellular CO2 partial pressure", "variable"],\
'c_KmC':        [35.9774,"-", "Scaling constan to adjust KmC with Temperature","constant"],\
'c_KmO':        [12.3772,"-", "Scaling constan to adjust KmC with Temperature","constant"],\
'c_Gamma_star': [11.187,"-", "Scaling constan to adjust KmC with Temperature","constant"],\
'c_Vcmax':      [26.355,"-", "Scaling constan to adjust Vcmax with Temperature","constant"],\
'CiH':          [None, "mu_bar", "Intercellular CO2 partial pressure under a high O2 condition", "variable"],\
'CiL':          [None, "mu_bar", "Intercellular CO2 partial pressure under under low O2 condition", "variable"],\
#'Ci_star':     [None, "mu_bar", "Ci-based CO2 compensation point in the absence of Rd", "parameter"],\
'CiH_star':     [None, "mu_bar", "Ci_star under a high O2 condition", "variable"],\
'CiL_star':     [None, "mu_bar", "Ci_star under low O2 condition", "variable"],\
'fcyc':         [0.0, "-", "Fraction of electrons at PSI that follow cyclic transport around PSI", "constant"],\
#'fpseudo':     [None, "-", "Fraction of electrons at PSI that follow pseudocyclic transport", "parameter"],\
#'fpseudo_b':   [None, "-", "Fraction of electrons at PSI that follow the basal pseudocyclic e- flow", "parameter"],\
'gm':           [None, "mol m**-2 s**-1 bar**-1", "Mesophyll diffusion conductance", "variable"],\
'gmo':          [None, "mol m**-2 s**-1 bar**-1", "Residual mesophyll diffusion conductance in the gm model (Eqn 11)", "variable"],\
#'h':           [None, "mol mol**-1", "Number of protons required to produce one ATP", "parameter"],\
'Iabs':         [None, "mu_mol photon m**-2 s**-1", "Photon flux density absorbed by leaf photosynthetic pigments", "variable"],\
'Iinc':         [None, "mu_mol photon m**-2 s**-1", "Photon flux density incident to leaves", "variable"],\
'J':            [None, "mu_mol e- m**-2 s**-1", "Linear plus additional pseudocyclic e- transport rate through PSII", "variable"],\
#'J′':          [None, "mu_mol e- m**-2 s**-1", "Rate of e- transport through PSII (applied in Eqn 3b)", "parameter"],\
'J2':           [None, "mu_mol e- m**-2 s**-1", "Total rate (linear plus basal and additional pseudocyclic) e- transport through PSII", "variable"],\
#'Jc':          [None, "mu_mol e- m**-2 s**-1", "Rate of e- transport calculated from CO2 uptake measurement", "parameter"],\
#'Jf':          [None, "mu_mol e- m**-2 s**-1", "Rate of e- transport calculated from the chlorophyll fluorescence measurement", "parameter"],\
'Jmax':         [None, "mu_mol e- m**-2 s**-1", "Maximum value of J under saturated light", "parameter"],\
'J2max':        [None, "mu_mol e- m**-2 s**-1", "Maximum value of J2 under saturated light", "parameter"],\
'O':            [None, "mbar", "Oxygen partial pressure", "variable"],\
'OH':           [None, "mbar", "High oxygen partial pressure", "variable"],\
'OL':           [None, "mbar", "Low oxygen partial pressure", "variable"],\
'Rd':           [None, "mu_mol CO2 m**-2 s**-1", "Respiratory CO2 release in the dark", "variable"],\
'R':            [ 0.008314, "J K**-1 mmol**-1", "Universal gas constant","constant"],\
'OK':            [273.15, "K", "Additive constant to convert from Celsius to Kelvin","constant"],\
's':            [None, "-", "Slope", "parameter"],\
'Tp':           [None, "mu_mol m**-2 s**-1", "Rate of triose phosphate export from the chloroplast", "parameter"],\
#'Zi':          [None, "-", "Dummy variables in Eqn 13", "parameter"],\
'alfa2_LL':     [None, "mol e- (mol photon)**-1", "Quantum efficiency of PSII e- transport under strictly limiting light, on the combined PSI- and PSII-absorbed light (i.e. Iabs) basis", "variable"],\
'Beta':         [0.86, "-", "Absorptance by leaf photosynthetic pigments", "parameter"],\
'delta':        [None, "-", "A parameter in the gm model, defining Cc : Ci ratio at saturating light", "parameter"],\
'dHa_KmC':      [80.99, "-", "Enthalpie of activation for KmC", "constant"],\
'dHa_KmO':      [23.72, "-", "Enthalpie of activation for KmO", "constant"],\
'dHa_Gamma_star':[24.46, "-", "Enthalpie of activation for Gamma_star", "constant"],\
'k2':           [None, "mol e- (mol photon)**-1", "Conversion efficiency of incident light into J", "variable"],\
'k2_LL':        [None, "mol e- (mol photon)**-1", "k2 at the strictly limiting light", "variable"],\
'Theta':        [None, "-", "Convexity factor for response of J to Iinc", "parameter"],\
'Theta2':       [None, "-", "Convexity factor for response of J2 to Iabs", "parameter"],\
'Ro2':          [None, "-", "Proportion of Iabs partitioned to PSII", "variable"],\
'Phi1_LL':      [0.95, "mol e- (mol photon)**-1", "Quantum efficiency of PSI e- flow at the strictly limiting light level, on the PSI-absorbed light basis", "constant"],\
'Phi2':         [None, "mol e- (mol photon)**-1", "Quantum efficiency of PSII e- flow on PSII-absorbed light basis, usually assessed from the chlorophyll fluorescence measurements", "variable"],\
'Phi2_LL':      [0.80, "mol e- (mol photon)**-1", "phi2 at the strictly limiting light level", "parameter"],\
#'PhiCO2':       [None, "mol CO2 (mol photon)**-1", "Quantum efficiency of CO2 assimilation on the Iabs basis", "parameter"],\
#'PhiCO2_LL':   [None, "mol e- (mol photon)**-1", "phiCO2 at the strictly limiting light level", "parameter"],\
#'Gamma':        [None, "mu_bar", "Cc-or Ci-based CO2 compensation point in the presence of Rd", "parameter"],\
#'GammaH':       [None, "mu_bar", "Gamma under a high O2 condition", "parameter"],\
#'Gamma_starH':  [None, "mu_bar", "Gamma_star under under a hogh O2 condition", "parameter"],\
#'Gamma_starL':  [None, "mu_bar", "Gamma_star under a low O2 condition", "parameter"],\
}


# Diccionario con las mediciones experimentales
datos_jitomate = {
'e2':{'USDA_Species_Code' :[],
      'Genotype'          :['Genotype'],
      'Sample_ID'         :['Jitomate'],
      'Constant'          :'Ci',    
      'Date'              :['date'],
      'Time'              :['date'],
      'A'                 :[-0.395678423, 1.139375965, 2.539459148, 3.846125114,5.357918383,11.03196795,14.76660096,16.92640294,18.24290956,20.22010214,21.19974921,22.39058412],
      'Ci'                :[602.9409108, 592.0412172,581.9911935, 572.7195395,561.8168128,520.9091002,493.3553555, 476.938528,466.8367222, 458.5607971,449.5573963,444.159444 ],
      'Patm'              :[],
      'Qin'               :[21.19124794, 39.7613678, 60.50613022,78.52490997,101.2146149,249.8605499,500.8461914,751.3582764,999.3603516,1250.824219,1500.171265,1749.610596],
      'RHs'               :[],
      'Tleaf'             :[19.72901154, 19.76031303, 19.8794136,19.95979881,20.06403542,20.37490463,20.94115067,21.51297951,22.09475327,22.65408325,23.259058,23.76675606],
      'O2'                :[210.0, 210.0, 210.0, 210.0, 210.0,210.0,210.0,210.0,210.0,210.0,210.0,210.0],
      'Qabs'              :[],
      'Φ2'                :[0.741230292, 0.731425666, 0.72034788, 0.710983321, 0.701849576, 0.6122588894, 0.4459077873, 0.3390587879, 0.269138719, 0.2202671696, 0.1893166374,0.1621622277],
      'A_Corrected'       :[-0.395678423, 1.139375965, 2.539459148, 3.846125114,5.357918383,11.03196795,14.76660096,16.92640294,18.24290956,20.22010214,21.19974921,22.39058412],    
      'Ci_Corrected'      :[602.9409108, 592.0412172,581.9911935, 572.7195395,561.8168128,520.9091002,493.3553555, 476.938528,466.8367222, 458.5607971,449.5573963,444.159444 ],
      'QC'                :[]
  },
'e1':{'USDA_Species_Code' :[],
      'Genotype'          :[],
      'Sample_ID'         :[],
      'Constant'          :'Qin',        
      'Date'              :[],
      'Time'              :[],
      'A'                 :[-0.286702772,2.790614889,8.361938752,13.45600121,17.56441261,21.99898565,23.59905728,24.19403033,23.56836163],
      'Ci'                :[52.11176386,81.34939444,143.2626428,209.2114258,280.3250099,448.1181592,633.5691219,826.4324589,1025.893119],
      'Patm'              :[],
      'Qin'               :[1750, 1750, 1750, 1750, 1750, 1750, 1750, 1750, 1750],
      'RHs'               :[],
      'Tleaf'             :[23.89445305,23.83109283,23.75239372,23.69697571,23.70287323,23.68829727,23.71489334,23.67053223,23.72981644],
      'O2'                :[210.0,210.0,210.0,210.0,210.0,210.0,210.0,210.0,210.0],
      'Qabs'              :[],
      'Φ2'                :[],
      'A_Corrected'       :[-0.286702772,2.790614889,8.361938752,13.45600121,17.56441261,21.99898565,23.59905728,24.19403033,23.56836163],    
      'Ci_Corrected'      :[52.11176386,81.34939444,143.2626428,209.2114258,280.3250099,448.1181592,633.5691219,826.4324589,1025.893119],
      'QC'                :[]
}
}

datos = {
'e1' :{ 'name': 'Experimento 1',
        '@varconst': 'C',
        'I': [21.19124794, 39.7613678, 60.50613022,78.52490997,101.2146149,249.8605499,500.8461914,751.3582764,999.3603516,1250.824219,1500.171265,1749.610596],
        'C': [602.9409108, 592.0412172,581.9911935, 572.7195395,561.8168128,520.9091002,493.3553555, 476.938528,466.8367222, 458.5607971,449.5573963,444.159444 ],
        'O': [210.0, 210.0, 210.0, 210.0, 210.0,210.0,210.0,210.0,210.0,210.0,210.0,210.0],
        'T': [19.72901154, 19.76031303, 19.8794136,19.95979881,20.06403542,20.37490463,20.94115067,21.51297951,22.09475327,22.65408325,23.259058,23.76675606],
        'Iset': [20.0, 40.0, 60.0, 80.0, 100.0, 250.0, 500.0, 750.0, 1000.0, 1250.0, 1500.0, 1750.0],
        'A': [-0.395678423, 1.139375965, 2.539459148, 3.846125114,5.357918383,11.03196795,14.76660096,16.92640294,18.24290956,20.22010214,21.19974921,22.39058412],
        'Phi2a' : [0.741230292, 0.731425666, 0.72034788, 0.710983321, 0.701849576, 0.6122588894, 0.4459077873, 0.3390587879, 0.269138719, 0.2202671696, 0.1893166374,0.1621622277],
        'Phi2' : [0.696157801,0.740858779,0.727894645,0.728846753,0.721839747,0.592216568,0.431875551,0.337748197,0.269858257,0.230867635,0.190884498,0.16189253]
    },
'e2' :{ 'name': 'Experimento 2',
        '@varconst': 'I',
        'I': [1750, 1750, 1750, 1750, 1750, 1750, 1750, 1750, 1750],
        'C': [52.11176386,81.34939444,143.2626428,209.2114258,280.3250099,448.1181592,633.5691219,826.4324589,1025.893119],
        'O': [210.0,210.0,210.0,210.0,210.0,210.0,210.0,210.0,210.0],
        'T': [23.89445305,23.83109283,23.75239372,23.69697571,23.70287323,23.68829727,23.71489334,23.67053223,23.72981644],
        'Iset': [1750,1750,1750,1750,1750,1750,1750,1750,1750],
        'A': [-0.286702772,2.790614889,8.361938752,13.45600121,17.56441261,21.99898565,23.59905728,24.19403033,23.56836163]
    }
}
