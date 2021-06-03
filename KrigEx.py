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

def genKrig(df, fmtn, xcol, ycol, zcol, grd_step):
    df_fmtn = df[df['fmtn']==fmtn].reset_index()
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
    df_krig = df_krig.dropna()
    x=np.array(df_krig[xcol]*grd_step)
    y=np.array(df_krig[ycol]*grd_step)
    z=np.array(df_krig[zcol])

    
    OK = OrdinaryKriging(
        x,
        y,
        z,
        variogram_model="linear",
        verbose=False,
        enable_plotting=False,
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


df = pd.read_csv('data/proc_combined3.csv')

Krig_var = 'TrueTemp'
dvn_map = genKrig(df, 
                 fmtn = 'DVN', 
                 xcol = 'CLS_Long', 
                 ycol='CLS_Lat', 
                 zcol = Krig_var,
                 grd_step=100)

egb_map = genKrig(df, 
                 fmtn = 'EGB', 
                 xcol = 'CLS_Long', 
                 ycol='CLS_Lat', 
                 zcol = Krig_var,
                 grd_step=100)

df_dvn = dvn_map['df_fmtn']
df_egb = egb_map['df_fmtn']

df_combine = pd.concat([df_dvn, df_egb])
df_combine[['UWI', Krig_var]].to_csv(f'data/Krig_{Krig_var}.csv')
