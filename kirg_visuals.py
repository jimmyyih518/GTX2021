# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 22:57:39 2021

@author: JZ2018
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

with open('Krig_Dict.pickle', 'rb') as handle:
    krig_dict = pickle.load(handle)
    
kvars = ['CLS_GAMMA', 'CLS_MUDTEMPC', 'CLS_RESISTIVITY', 'CLS_POROSITY', 
         'CLS_TrueTempGrad_CpM', 'EX_TrueTemp', 'EX_TrueTempGrad_CpM',
         'CLS_CumOil_bbl', 'CLS_CumGas_mcf', 'CLS_CumWater_bbl', 'CLS_OGR_bblmmcf']    
fmtn = 'egb'
for k in range(0, len(kvars)):
    krig_var = kvars[k]
    zmap = krig_dict[f'{fmtn}_{krig_var}']['zmap']
    df_fmtn = krig_dict[f'{fmtn}_{krig_var}']['df_fmtn']
    ycol = 'Lat'
    xcol = 'Long'
    
    grd_step=100
    gmin_lat = int(df_fmtn[ycol].min()-1)
    gmax_lat = int(df_fmtn[ycol].max()+1)
    gmin_lon = int(df_fmtn[xcol].min()-1)
    gmax_lon = int(df_fmtn[xcol].max()+1)
    #grd_step = 100
        
    gridy = list(range(gmin_lat*grd_step,gmax_lat*grd_step))
    gridy = [float(i) for i in gridy]
    gridx = list(range(gmin_lon*grd_step,gmax_lon*grd_step))
    gridx = [float(i) for i in gridx]
    
    zmap_df = pd.DataFrame(zmap)
    zmap_df.columns = [int(x) for x in gridx]
    zmap_df.index = [int(y) for y in gridy]
    zmapt = zmap_df.sort_index(ascending=False)
    grp_df = df_fmtn.groupby('Set')
    
    plt.imshow(zmap, alpha=0.5)
    for name, group in grp_df:
        plt.scatter((group['Long']-gmin_lon)*100, (gmax_lat-group['Lat'])*100,marker='o', label=name)
    plt.legend()
    plt.contour(zmap, colors='black')
    plt.title(f'Eaglebine Mapped {krig_var}')
    plt.show()


#plt.scatter((dfmap['Long']-gmin_lon)*100, (dfmap['Lat']-gmin_lat)*100, colors=dfmap['Set'])
df_mod = pd.read_csv('data/mod_data.csv')
df_mod['EX_TrueTempGrad_CpM'] = df_mod['EX_TrueTemp'] / df_mod['TVDMSS']
df_dvn = df_mod[df_mod['fmtn']=='DVN']
df_egb = df_mod[df_mod['fmtn']=='EGB']


import seaborn as sns
sns.scatterplot(df_dvn['Long'],  df_dvn['Lat'], hue=df_dvn['EX_TrueTemp'], size = df_dvn['EX_TrueTemp'])
plt.title('Duvernay Well Locations and True Formation Temp (C)')
sns.scatterplot(df_egb['Long'],  df_egb['Lat'], hue=df_egb['EX_TrueTemp'], size = df_egb['EX_TrueTemp'])
plt.title('Eaglebine Well Locations and True Formation Temp (C)')

sns.scatterplot(df_egb['Long'],  df_egb['Lat'], hue=df_egb['Set'])
plt.title('Eaglebine Input')
sns.scatterplot(df_dvn['Long'],  df_dvn['Lat'], hue=df_dvn['Set'])
plt.title('Duvernay Input')




labelcol = 'EX_TrueTempGrad_CpM'
labelcol = 'EX_TrueTemp'
feature_cols = [ 'TVDMSS', 'Lat', 'Long', 'DST_TempGrad_CpM',
                'DST_TVDMSS','DST_TempC'] + ['Krig_'+ x for x in kvars]
df_dvn_plot = df_dvn[[labelcol]+feature_cols+['UWI','Set', 'fmtn']]
df_egb_plot = df_egb[[labelcol]+feature_cols+['UWI', 'Set','fmtn']]

plt.hist(df_dvn_plot[labelcol], alpha=0.5, label='DVN')
plt.hist(df_egb_plot[labelcol], alpha = 0.5, label='EGB')
plt.xlabel('Temperature Gradient C/meter')
plt.ylabel('Frequency')
plt.axvline(0.03219, label = 'DVN Temp Gradient C/M', c='black')
plt.axvline(0.0365, label = 'EGB Temp Gradient C/M', c = 'red')
plt.legend()

feature_cols = ['Lat', 'Long', 'TVDMSS', 'DST_TempC',  'Krig_CLS_POROSITY', 
                'Krig_EX_TrueTemp', 'Krig_EX_TrueTempGrad_CpM', 
                'Krig_CLS_CumWater_bbl', 'Krig_CLS_CumGas_mcf']

sns.pairplot(df_dvn_plot, y_vars = labelcol, x_vars = df_dvn_plot[[labelcol]+feature_cols].columns.values)
df_dvn_plot = df_dvn[[labelcol]+feature_cols]
g=sns.PairGrid(df_dvn_plot)
g.map(sns.scatterplot)


sns.scatterplot(df_mod['TVDMSS'],df_mod[labelcol], hue=df_mod['fmtn'])
sns.scatterplot(df_mod['DST_TempC'],df_mod[labelcol], hue=df_mod['fmtn'])
df_mod['Log10_OGR_bblmmcf'] = np.log10(df_mod['Krig_CLS_OGR_bblmmcf'])
sns.scatterplot(df_mod['Log10_OGR_bblmmcf'],df_mod[labelcol], hue=df_mod['fmtn'])
