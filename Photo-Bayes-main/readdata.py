#%%
import pandas as pd
import matplotlib.pyplot as plt


# Funcion que toma el dataframe, 
# el sampleID y el valor de Oxigeno y
# regresa el diccionario del experimento

def experiment2dict(df,sampleID,O2val,constant):
    '''
    Del dataframe creo el diccionario de 
    un experimento
    '''    
    colnames = df.columns
    # Filtros por sampleID y O2val
    df1 = df[df[colnames[2]] == sampleID] 
    df2 = df1[df1['O2'] == O2val]
    
    # Tomo los titulos y los valores
    cnames = df2.columns.values.tolist()
    cvalues = df2.T.values.tolist()
    
    # Elimino duplicados que identifican el experimento
    cvalues2 =[]
    for rows in cvalues[0:4]:
        cvalues2.append([*set(rows)])
 
    #Añado que es a Qin constante
    cnames.insert(3,'Constant')  
    cvalues2.insert(3 ,constant)
    # Junto las claves que identifican con los datos 
    cvalues2.extend(cvalues[4:])
    # Creo el diccionario final con las llaves
    dic=dict(zip(cnames, cvalues2))

    #Regreso el diccionario
    return dic

# Leo el excell con los datos 

def datosendic( oxigeno = [210], N = 2):
    #Leo datos y nombres de muestras
    df = pd.read_excel('C:\\Users\\uriel\\Downloads\\Photo-Bayes-main-20250303T072921Z-001\\Photo-Bayes-main\\Photo-Bayes-main\\data\\ACi-Curves.xlsx', header=[0], sheet_name = 'ACi-Curves') #'./data/ACi-Curves.xlsx'
    colnames = df.columns
    samples=df[colnames[2]].unique()
    # Considero experimentos a dos niveles de oxigeno
    #oxigeno = [21]

    k = 1 #Indice para los experimentos

    dicionario = {}
    # Loop sobre la planta 
    for sampleID in samples[N:(N+1)]:
    # Loop para alto y bajo oxigeno
        for Ox2 in oxigeno:
            expNo = 'e' + str(k)
            dic = experiment2dict(df,sampleID,Ox2,constant ='Qin')
            print(expNo + ' : Q ctt ' + ' : ' +dic['Sample_ID'][0]+' @ O2 = '+str(dic['O2'][0]))
            dicionario[expNo] = dic
            k = k + 1

    ####################################################
    #Leo datos y nombres de muestras
    df2 = pd.read_excel('C:\\Users\\uriel\\Downloads\\Photo-Bayes-main-20250303T072921Z-001\\Photo-Bayes-main\\Photo-Bayes-main\\data\\AQ-Curves.xlsx', 
             header=[0], 
             sheet_name = 'AQ-Curves')

    colnames2 = df2.columns
    samples2=df2[colnames2[2]].unique()

    for sampleID in samples2[N:(N+1)]:
       # Loop para alto y bajo oxigeno
       for Ox2 in oxigeno:
            expNo = 'e' + str(k)
            dic = experiment2dict(df2,sampleID,Ox2,constant ='Ci')
            print(expNo + ' : Ci ctt ' + ' : ' +dic['Sample_ID'][0]+' @ O2 = '+str(dic['O2'][0]))
            dicionario[expNo] = dic
            k = k + 1

    return dicionario        
#%%

if __name__ == "__main__":

    datos = datosendic()













# %%
