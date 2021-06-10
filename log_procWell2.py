# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 16:39:55 2021

@author: JZ2018
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from helper_funs import *

##----------------------------------------
df_log = pd.read_parquet('data/log_processed_all.parquet')
##----------------------------------------
set_assign = pd.read_csv('data/set_assign.csv')
static_logs = pd.read_csv('data/Data_static_logs.csv')
static_logs.columns = ['UWI', 'SLOG_TVDFT', 'SLOG_TEMPC', 'fmtn']
static_logs['SLOG_TVDM'] = static_logs['SLOG_TVDFT']/3.28084
#static_logs['SLOG_TempGrad_CpM'] = static_logs['SLOG_TEMPC'] / static_logs['SLOG_TVDM']
##----------------------------------------
egb_head = pd.read_excel('data/Eaglebine/Eaglebine well headers SPE April 21 2021.xlsx')
egb_dst = pd.read_excel('data/Eaglebine/Eaglebine BHT TSC data for SPE April 21 2020.xlsx')
egb_prod = pd.read_excel('data/Eaglebine/SPE Eaglebine production summary April 20 2021.xlsx')
egb_tt = pd.read_excel('data/Eaglebine/Eaglebine TrueTemp_Train2.xlsx')
##----------------------------------------
dvn_head = pd.read_excel('data/Duvernay/Duvernay well headers SPE April 21 2021 .xlsx')
dvn_dst = pd.read_excel('data/Duvernay/Duvernay DST BHT for SPE April 20 2021.xlsx', sheet_name='DST BHT')
dvn_prod = pd.read_excel('data/Duvernay/SPE Duvernay production summary April 20 2021.xlsx')
dvn_tt = pd.read_excel('data/Duvernay/Duvenay TrueTemp_Train.xlsx')
##----------------------------------------
#wellcols = ['UWI', 'Lat', 'Long', 'TVDMSS', 'TVDMKB','TrueTemp', 'TempGrad']
##----------------------------------------
dvn_head = dvn_head[['UWI ', 'Elevation Meters', 'BottomLatitude_NAD27', 'BottomLongitude_NAD27']]
dvn_head.columns = ['UWI', 'ELEV_M', 'Lat', 'Long']
dvn_prod['TVDMKB'] = dvn_prod['Total Vertical Depth (ft)   ']/3.28084
dvn_prod = dvn_prod[['API   ', 'TVDMKB']]
dvn_prod.columns = ['UWI', 'TVDMKB']
dvn_combine = dvn_head.merge(dvn_prod, how = 'left', on='UWI')
dvn_combine['TVDMSS'] = dvn_combine['TVDMKB'] - dvn_combine['ELEV_M']
dvn_dst.columns = ['UWI', 'DST_TVDMSS', 'DST_TempC']
dvn_combine = dvn_combine.merge(dvn_dst, how='left', on='UWI')
dvn_tt.columns = ['UWI', 'TT_TVDMSS', 'TrueTemp']
zbins = [x*20 for x in range(-100,200)]
dvn_tt['BinDepth'] = pd.cut(dvn_tt['TT_TVDMSS'], zbins)
dvn_combine['BinDepth'] = pd.cut(dvn_combine['TVDMSS'], zbins)
dvn_combine2 = dvn_combine.merge(dvn_tt, how='left', on=['UWI', 'BinDepth'])
dvn_combine2 = dvn_combine2.merge(set_assign, how='left', on='UWI')
dvn_combine2 = dvn_combine2.merge(static_logs, how='left', on='UWI')
dvn_combine2['TrueTempGrad_CpM'] = dvn_combine2['TrueTemp'] / dvn_combine2['TT_TVDMSS']
dvn_combine2['SLOG_TempGrad_CpM'] = dvn_combine2['SLOG_TEMPC'] / (dvn_combine2['SLOG_TVDM'] - dvn_combine2['ELEV_M'])
dvn_combine2['CLS_TrueTempGrad_CpM'] = coalesce(dvn_combine2, ['SLOG_TempGrad_CpM', 'TrueTempGrad_CpM'])
dvn_combine2['DST_TempGrad_CpM'] = dvn_combine2['DST_TempC'] / (dvn_combine2['DST_TVDMSS'])
dvn_combine2 = dvn_combine2.drop_duplicates(subset=['UWI'])
plt.scatter(dvn_combine2['DST_TempGrad_CpM'], dvn_combine2['CLS_TrueTempGrad_CpM'])
dvn_combine2['fmtn'] = 'DVN'

##----------------------------------------
egb_head['ELEV_M'] = egb_head['Elevation']/3.28084
egb_head = egb_head[['displayapi', 'ELEV_M', 'BottomLatitude_NAD27', 'BottomLongitude_NAD27']]
egb_head.columns = ['UWI', 'ELEV_M', 'Lat', 'Long']
egb_prod['TVDMKB'] = egb_prod['Total Vertical Depth (ft)   ']/3.28084
egb_prod = egb_prod[['API   ', 'TVDMKB']]
egb_prod.columns = ['UWI', 'TVDMKB']
egb_combine = egb_head.merge(egb_prod, how='left', on='UWI')
egb_combine['TVDMSS'] = egb_combine['TVDMKB'] - egb_combine['ELEV_M']
egb_dst['TVDMKB'] = egb_dst['BHT_ subsurface (ft)'] / 3.28084
egb_dst['TVDMSS'] = egb_dst['BHT_below sea level (ft)'] / 3.28084
egb_dst['DST_TempC'] = (egb_dst['BHTorMRT (maximum recorded temperature) oF'] -32) *5 / 9
egb_dst = egb_dst[['UWI', 'TVDMSS', 'DST_TempC', 'SurfLat', 'SurfLong']]
egb_dst.columns = ['UWI', 'DST_TVDMSS', 'DST_TempC', 'SurfLat', 'SurfLong']
egb_combine = egb_combine.merge(egb_dst, how='left', on='UWI')
egb_tt.columns = ['UWI', 'TVDSSFT', 'TrueTemp']
egb_tt['TVDMSS'] = egb_tt['TVDSSFT'] / 3.28084
egb_tt['TrueTemp'] = (egb_tt['TrueTemp']-32)*5/9
egb_tt['BinDepth'] = pd.cut(egb_tt['TVDMSS'], zbins)
egb_tt = egb_tt.drop('TVDSSFT', axis=1)
egb_tt.columns = ['UWI', 'TrueTemp', 'TT_TVDMSS', 'BinDepth']
egb_combine['BinDepth'] = pd.cut(egb_combine['TVDMSS'], zbins)
egb_combine2 = egb_combine.merge(egb_tt, how='left', on=['UWI', 'BinDepth'])
egb_combine2['UWI'] = egb_combine2['UWI'].astype(str)
egb_combine2 = egb_combine2.merge(set_assign, how='left', on='UWI')
egb_combine2 = egb_combine2.merge(static_logs, how='left', on='UWI')
#egb_combine2['TT_TVDMKB'] = egb_combine2['TT_TVDMSS'] + egb_combine2['ELEV_M']
egb_combine2['Lat'] = coalesce(egb_combine2, ['Lat', 'SurfLat'])
egb_combine2['Long'] = coalesce(egb_combine2, ['Long', 'SurfLong'])
egb_combine2['TrueTempGrad_CpM'] = egb_combine2['TrueTemp'] / egb_combine2['TT_TVDMSS']
egb_combine2['SLOG_TempGrad_CpM'] = egb_combine2['SLOG_TEMPC'] / (egb_combine2['SLOG_TVDM'] - egb_combine2['ELEV_M'])
egb_combine2['CLS_TrueTempGrad_CpM'] = coalesce(egb_combine2, ['SLOG_TempGrad_CpM', 'TrueTempGrad_CpM'])
egb_combine2['DST_TempGrad_CpM'] = egb_combine2['DST_TempC'] / (egb_combine2['DST_TVDMSS'])
egb_combine2 = egb_combine2.drop_duplicates(subset=['UWI'])
plt.scatter(egb_combine2['DST_TempGrad_CpM'], egb_combine2['CLS_TrueTempGrad_CpM'])
egb_combine2['fmtn'] = 'EGB'
egb_combine2 = egb_combine2.drop(['SurfLat', 'SurfLong'], axis=1)
##----------------------------------------
outcombine = pd.concat([dvn_combine2, egb_combine2])
plt.scatter(outcombine['TVDMSS'], np.log10(outcombine['CLS_TrueTempGrad_CpM']))
plt.hist(np.log10(outcombine['DST_TempGrad_CpM']))
#plt.scatter(outcombine[''])
np.setdiff1d(egb_combine2.columns.tolist(), dvn_combine2.columns.tolist())
np.setdiff1d( dvn_combine2.columns.tolist(), egb_combine2.columns.tolist())
outcombine.to_csv('data/proc_outcombine.csv')
##----------------------------------------
#df_well =pd.read_csv('Combined3a.csv')
# df_well['CLS_ELEV_M'] = np.where(df_well['fmtn']=='DVN', 
#                                  pd.to_numeric(df_well['Elevation Meters_0'], errors='coerce'),
#                                  pd.to_numeric(df_well['Elevation(f)_0'], errors='coerce')/3.28084)

# wellcols = ['UWI', 'BHT', 'TrueTemp',  'fmtn',
#             'CLS_Lat', 'CLS_Long', 'CLS_Depth_ft', 'CLS_ELEV_M', 'Krig_Temp']

# df_ldvn=pd.read_parquet('data/log_processed_dvn.parquet')
# df_legb=pd.read_parquet('data/log_processed_egb.parquet')

# df = pd.concat([df_ldvn, df_legb])

# df_log = df_log.reset_index(drop=True)
# #df1 = df[df['UWI']==df['UWI'][0]]
# # df1 = df_log.copy()
# #df1 = df[df['UWI']=='102112204416W500']
# # df1a = df1.copy()
# depth_bins = [x*25 for x in range(0, 1000)]
# df_log['BinDepth'] = pd.cut(df_log['CLS_DEPTH_M'], depth_bins)

# df_log2 = df_log.groupby(['UWI', 'BinDepth']).agg({'CLS_DEPTH_M':'mean',
#                                             'CLS_GAMMA':'mean',
#                                            'CLS_MUDTEMPC':'mean',
#                                            'CLS_RESISTIVITY':'mean',
#                                            'CLS_POROSITY':'mean'}).dropna(
#                                                subset=['CLS_GAMMA','CLS_MUDTEMPC','CLS_RESISTIVITY','CLS_POROSITY'], how='all').reset_index()

# df_log2['BinDepth'] = df_log2['BinDepth'].astype(str)
# df_log2.to_parquet('data/log_processed_all.parquet')
                                               
df2 = df_log.merge(outcombine, how='left', left_on='UWI', right_on='UWI')
#df2['CLS_BHT_DEPTH_M'] = df2['CLS_Depth_ft']/3.28084
df2['depth_diff'] = abs(df2['TVDMSS'] - df2['CLS_DEPTH_M'])
df2 = df2.sort_values(by=['UWI', 'depth_diff'])
df2a = df2.groupby(['UWI']).first().reset_index()
df2b = df2a[['UWI', 'CLS_DEPTH_M', 'CLS_GAMMA', 'CLS_MUDTEMPC',
       'CLS_RESISTIVITY', 'CLS_POROSITY']]

df2c = outcombine.merge(df2b, how='left', on ='UWI')
df2c.to_csv('data/proc_log_well_combine.csv')
