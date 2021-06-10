# -*- coding: utf-8 -*-
"""
Created on Fri May 28 16:49:14 2021

@author: JZ2018
"""

import pandas as pd
import numpy as np
import seaborn as sns
from tqdm.notebook import tqdm


dvn_bht = pd.read_excel('data/Duvernay/Duvernay DST BHT for SPE April 20 2021.xlsx')
dvn_truth = pd.read_excel('data/Duvernay/Duvenay TrueTemp_Train.xlsx')
dvn_head = pd.read_excel('data/Duvernay/Duvernay well headers SPE April 21 2021 .xlsx')
static_logs = pd.read_csv('data/Data_static_logs.csv')

UWI = '100061706804W600'

df_eda = dvn_truth[dvn_truth['UWI']==UWI]
df_eda = df_eda.merge(dvn_head, how = 'left', left_on='UWI', right_on='UWI ')
df_eda['tvd_mkb'] = df_eda['Depths subsea (m)'] + df_eda['Elevation Meters']
#test = dvn_truth.copy()
df_eda['temp_grad'] = df_eda['True Temperature (oC)'] / df_eda['Depths subsea (m)']
#df_eda['UWI'] = test['UWI'].astype(str)
#ax = sns.scatterplot(data=df_eda[df_eda['tvd_mkb']>0], x='temp_grad',y='tvd_mkb',  hue='UWI')
ax = sns.scatterplot(data=df_eda[df_eda['Depths subsea (m)']>0], x='temp_grad',y='Depths subsea (m)')
ax.invert_yaxis()
ax.set_xscale('log')
ax.set_yscale('log')

from sklearn.linear_model import LinearRegression
lin_mods = {}
for uwi in test['UWI'].unique():
    sub_test = test[test['UWI']==uwi]
    sub_test = sub_test[sub_test['Depths subsea (m)']>0]
    sub_test['x_sq'] = sub_test['Depths subsea (m)'] ** 2
    X=sub_test[['Depths subsea (m)','x_sq']]
    y=sub_test['True Temperature (oC)']
    mod = LinearRegression()
    mod.fit(X, y)
    # mod.score(X, y)
    # mod.intercept_
    # mod.coef_
    lin_mods[uwi] = mod
    
import matplotlib.pyplot as plt
plot_df = df_eda[df_eda['Depths subsea (m)']>0]
plot_df2 = dvn_bht[dvn_bht['UWI']==UWI]
plot_df2['temp_grad'] = plot_df2['DST Bottom Hole Temp. (degC)'] / plot_df2['DST End Depth (MD) (m)']
plot_df2['depth_subsea'] = plot_df2['DST End Depth (MD) (m)'] - plot_df2['elevation M above sea level']
plot_df3 = static_logs[static_logs['Well_ID'] == UWI]

plot_df3=plot_df3.merge(plot_df2, how='left', left_on='Well_ID', right_on='Well ID')
plot_df3['depth_subsea'] = plot_df3['Depth (ft)'] / 3.28084 - plot_df3['elevation M above sea level']
plot_df3['temp_grad'] = plot_df3['Temp (degC)'] / plot_df3['depth_subsea']

ax = plt.gca()
ax.scatter(plot_df['temp_grad'], plot_df['Depths subsea (m)'], color='red')
ax.scatter(plot_df2['temp_grad'], plot_df2['depth_subsea'], color='green')
ax.scatter(plot_df3['temp_grad'], plot_df3['depth_subsea'], color='blue')
ax.invert_yaxis()
#ax.set_yscale('log')
ax.set_xscale('log')
