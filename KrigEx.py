# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 22:50:07 2021

@author: JZ2018
"""

import pandas as pd
import numpy as np
import pykrige.kriging_tools as kt
from pykrige.ok import OrdinaryKriging
import matplotlib.pyplot as plt
from helper_funs import *

def genKrig(df, fmtn, xcol, ycol, zcol, grd_step):
    df_fmtn = df[df['fmtn']==fmtn].reset_index()
    #df_fmtn = df_fmtn.dropna(subset=[xcol, ycol, zcol])
    gmin_lat = int(df_fmtn[ycol].min()-1)
    gmax_lat = int(df_fmtn[ycol].max()+1)
    gmin_lon = int(df_fmtn[xcol].min()-1)
    gmax_lon = int(df_fmtn[xcol].max()+1)
    #grd_step = 100
    
    gridy = list(range(gmin_lat*grd_step,gmax_lat*grd_step))
    gridy = [float(i) for i in gridy]
    gridx = list(range(gmin_lon*grd_step,gmax_lon*grd_step))
    gridx = [float(i) for i in gridx]
    

    df_krig = df_fmtn[[xcol, ycol, zcol]]
    #df_krig[zcol] = np.where(df_krig[zcol]>190, None, df_krig[zcol])
    df_krig = df_krig.dropna()
    x=np.array(df_krig[xcol]*grd_step)
    y=np.array(df_krig[ycol]*grd_step)
    z=np.array(df_krig[zcol])
    
    std_min = np.std(z)

    
    OK = OrdinaryKriging(
        x,
        y,
        z,
        variogram_model="linear",
        verbose=False,
        enable_plotting=False
    )
    
    z, ss = OK.execute("grid", gridx, gridy)
    
    #if np.std(z) <= std_min*0.1:
    if True:
        z=np.array(df_krig[zcol])
        
        OK = OrdinaryKriging(
            x,
            y,
            z,
            variogram_model="gaussian", 
            variogram_parameters={'sill': (max(z)-min(z))*0.9, 
                                 'range': 0.8*((max(x)-min(x))**2 + (max(y)-min(y))**2)**0.5, 
                                 'nugget': (max(z)-min(z))*0.05},
            exact_values=False
        )
        z, ss = OK.execute("grid", gridx, gridy)
    
    df_map = pd.DataFrame(z)
    df_map.columns = [int(x) for x in gridx]
    df_map.index = [int(y) for y in gridy]
    #kt.write_asc_grid(gridx, gridy, z, filename="output.asc")
    #plt.imshow(z)
    #plt.show()
    #plt.scatter(x,y)


    wlist = []
    for w in range(0,df_fmtn.shape[0]):
        #print(w)
        wx = int(df_fmtn[xcol][w]*grd_step)
        wy = int(df_fmtn[ycol][w]*grd_step)
        wval = df_map[wx][wy]
        wlist.append(wval)

    Kcol = f'Krig_{zcol}'
    df_fmtn[Kcol] = wlist    
    #df_fmtn = df_fmtn[[xcol, ycol, zcol, Kcol]]
    #plt.scatter(Combined3_dvn['Krig_Temp'], Combined3_dvn['TrueTemp'])
    #plt.plot([0, 110], [0, 110], label='1-1', c='k')
    #Combined3_dvn.to_csv('Combined3_dvn_krig.csv')
    
    out_dict = {}
    out_dict['df_fmtn'] = df_fmtn
    out_dict['zmap'] = z
    out_dict['ss'] = ss
    
    return out_dict



#df = pd.read_csv('data/proc_combined3.csv')
#df = pd.read_csv('data/proc_log_well_combine.csv')
#Combined3 = pd.read_csv('Combined3.csv')
#ex_df = Combined3[['UWI', 'TrueTemp']]
#ex_df.columns = ['UWI', 'EX_TrueTemp']
#df = df.merge(ex_df, how='left', on='UWI')
#df['EX_TrueTempGrad_CpM'] = df['EX_TrueTemp'] / df['TVDMSS']

df =   pd.read_csv('data/Final_Preds.csv')
df['NNET_Error'] = df['EX_TrueTemp'] - df['NNET_Pred_TempC']
##----------------------------------------------------------
Combined3['CLS_CumOil_bbl'] = coalesce(Combined3, ['Oil Total Cum (bbl)_0','Oil Total Cum (bbl)_1','Oil Total Cum (bbl)_2','Oil Total Cum (bbl)_3',
                                                   'Oil Maximum (bbl)_0', 'Oil Maximum (bbl)_1', 'Oil Maximum (bbl)_2', 'Oil Maximum (bbl)_3',
                                                   'Oil Total Cum (bbl)_0.1', 'Oil Maximum (bbl)_0.1'])
Combined3['CLS_CumGas_mcf'] = coalesce(Combined3, ['Gas Total Cum (mcf)_0','Gas Total Cum (mcf)_1','Gas Total Cum (mcf)_2','Gas Total Cum (mcf)_3',
                                                   'Gas Maximum (mcf)_0', 'Gas Maximum (mcf)_1', 'Gas Maximum (mcf)_2', 'Gas Maximum (mcf)_3',
                                                   'Gas Total Cum (mcf)_0.1', 'Gas Maximum (mcf)_0.1'])
Combined3['CLS_CumWater_bbl'] = coalesce(Combined3, ['Water Total Cum (bbl)_0','Water Total Cum (bbl)_1','Water Total Cum (bbl)_2','Water Total Cum (bbl)_3',
                                                   'Water Maximum (bbl)_0', 'Water Maximum (bbl)_1', 'Water Maximum (bbl)_2', 'Water Maximum (bbl)_3',
                                                   'Water Total Cum (bbl)_0.1', 'Water Maximum (bbl)_0.1'])

Combined3['CLS_OGR_bblmmcf'] = Combined3['CLS_CumOil_bbl'] / Combined3['CLS_CumGas_mcf']

df_prod = Combined3[['UWI', 'CLS_CumOil_bbl', 'CLS_CumGas_mcf', 'CLS_CumWater_bbl', 'CLS_OGR_bblmmcf']]
df = df.merge(df_prod, how='left', on='UWI')
df.replace([np.inf, -np.inf], np.nan, inplace=True)
##----------------------------------------------------------
kvars = ['CLS_GAMMA', 'CLS_MUDTEMPC', 'CLS_RESISTIVITY', 'CLS_POROSITY', 
         'CLS_TrueTempGrad_CpM', 'EX_TrueTemp', 'EX_TrueTempGrad_CpM',
         'CLS_CumOil_bbl', 'CLS_CumGas_mcf', 'CLS_CumWater_bbl', 'CLS_OGR_bblmmcf']

#Krig_var =  kvars[3]

kvars= ['NNET_Error']
krig_dict = {}
for Krig_var in kvars:
    print(Krig_var)
    dvn_map = genKrig(df, 
                     fmtn = 'DVN', 
                     xcol = 'Long', 
                     ycol='Lat', 
                     zcol = Krig_var,
                     grd_step=100)
    
    egb_map = genKrig(df, 
                     fmtn = 'EGB', 
                     xcol = 'Long', 
                     ycol='Lat', 
                     zcol = Krig_var,
                     grd_step=100)

    krig_dict[f'dvn_{Krig_var}'] = dvn_map
    krig_dict[f'egb_{Krig_var}'] = egb_map
    
    #plt.imshow(dvn_map['zmap'])
    #plt.imshow(egb_map['zmap'])

    df_dvn = dvn_map['df_fmtn']
    df_egb = egb_map['df_fmtn']

    df_combine = pd.concat([df_dvn, df_egb])

    df_combine[['UWI', f'Krig_{Krig_var}']].to_csv(f'data/Krig_{Krig_var}.csv')

plt.imshow(dvn_map['zmap'])
plt.imshow(egb_map['zmap'])

grp_df = df[df['fmtn']=='EGB'].groupby('Set')
ycol = 'Lat'
xcol = 'Long'
    
grd_step=100
gmin_lat = int(df[df['fmtn']=='EGB'][ycol].min()-1)
gmax_lat = int(df[df['fmtn']=='EGB'][ycol].max()+1)
gmin_lon = int(df[df['fmtn']=='EGB'][xcol].min()-1)
gmax_lon = int(df[df['fmtn']=='EGB'][xcol].max()+1)

plt.imshow(egb_map['zmap'], alpha=0.5)
for name, group in grp_df:
    plt.scatter((group['Long']-gmin_lon)*100, (gmax_lat-group['Lat'])*100,marker='o', label=name)
plt.legend()
plt.contour(egb_map['zmap'], colors='black')
plt.title(f'Mapped  Residuals')
plt.show()



# import pickle
# pickle.dump(krig_dict, open('data/Krig_Dict.pickle', 'wb'))
#krig_dict = pd.read_pickle(r'data/Krig_Dict.pickle')
#sns.scatterplot(df_fmtn['Long'], df_fmtn['Lat'], hue=df_fmtn['CLS_GAMMA'])
with open('Krig_Dict.pickle', 'wb') as handle:
    pickle.dump(krig_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('Krig_Dict.pickle', 'rb') as handle:
    b = pickle.load(handle)