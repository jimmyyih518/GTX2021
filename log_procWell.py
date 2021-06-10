# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 00:26:55 2021

@author: JZ2018
"""

import pandas as pd
#df_log = pd.read_csv('data/combine_log.csv')

class renamer():
    def __init__(self):
        self.d = dict()
    def __call__(self, x):
        if x not in self.d:
            self.d[x] = 0
            return x
        else:
            self.d[x] += 1
            return "%s_%d" % (x, self.d[x])


def coalesce(df, column_names):
    i = iter(column_names)
    column_name = next(i)
    answer=df[column_name]
    for column_name in i:
            answer = answer.fillna(df[column_name])
    return answer


df_log = pd.DataFrame()
for i in range(0,4):
    print(i)
    temp_df = pd.read_parquet(f'data/egbtemplogs/temp_combine_log{i}.parquet')
    temp_df.columns = [' '.join(s.split()) for s in temp_df.columns]
    temp_df = temp_df.rename(columns=renamer())
    df_log=pd.concat([df_log.reset_index(drop=True), temp_df.reset_index(drop=True)])
    del temp_df

df_log.to_parquet('data/combine_log_egb.parquet')

df_log_dvn = pd.read_parquet('data/combine_log_dvn.parquet')
df_log_egb = pd.read_parquet('data/combine_log_egb.parquet')

#df_log = pd.concat([df_log1, df_log2])
df_log = df_log_egb

##------------------------------------------------
#df_log=pd.read_parquet('data/combine_log.parquet')
#df_log = pd.read_parquet('data/combine_log_dvn.parquet')
df_well =pd.read_csv('data/proc_combined3.csv')

# all_cols = df_log.columns.tolist()
# mis_cols = df_log.isna().sum().reset_index()

#uwi_log = df_log['UWI'][1]
#uwi_well = df_well['UWI'][1]
df_log['UWI'] = df_log['UNIQUE WELL ID'].str.replace('. ','')+'0'
#df_well_f = df_well[df_well['UWI']==uwi_log]
#df_log_filt = df_log[df_log['UWI']==uwi_log]

#wellcols = df_well.columns.tolist()
wellcols = ['UWI', 'BHT', 'TrueTemp', 'Depth_SS(m)', 'fmtn',
            'CLS_Lat', 'CLS_Long', 'CLS_Depth_ft', 'Krig_Temp']

df_well = df_well[wellcols]
#logcols = df_log.columns.tolist()
#logcols = ['UWI',]

find_cols = [col for col in df_log.columns if 'GAMMA' in col.upper()]
df_log['CLS_GAMMA'] = coalesce(df_log, find_cols)
df_gamma = pd.DataFrame(df_log['CLS_GAMMA'])
df_gamma['CLS_GAMMA'] = pd.to_numeric(df_gamma['CLS_GAMMA'], errors='coerce')
import numpy as np
df_gamma['CLS_GAMMA'] = np.where(df_gamma['CLS_GAMMA']<=-999, None, df_gamma['CLS_GAMMA'])
df_gamma['CLS_GAMMA'] = np.where(df_gamma['CLS_GAMMA']<0,0,df_gamma['CLS_GAMMA'])
df_gamma['CLS_GAMMA'] = np.where(df_gamma['CLS_GAMMA']>200, 200, df_gamma['CLS_GAMMA'])

import matplotlib.pyplot as plt
plt.hist(df_gamma['CLS_GAMMA'].dropna())

# find_cols = [col for col in df_log.columns if 'DEPTH' in col.upper()]
# find_cols = ['Depth Curve', 'Depth Curve_1','DEPTH', '1 MEASURED DEPTH', 'DEPTH FEET', 
#               'DEPTH FEET_1',  'Depth Feet',
#              'DEPTH METER', 'DEPTH.FT']
# find_cols = [
#              'Depth METER', 'DEPTH.M', 'Depth meter'
#              ]


#df_log['DEPTH METER'] = pd.to_numeric(df_log['DEPTH METER']) / 3.28084
#df_log['depth_ft'] = coalesce(df_log, find_cols)
#df_log['depth_m'] = coalesce(df_log, find_cols)

# df_log['CLS_DEPTH_M'] = np.where(pd.to_numeric(df_log['depth_m']).isna(), 
#                                  pd.to_numeric(df_log['depth_ft'])/3.28084, 
#                                  pd.to_numeric(df_log['depth_m']))


df_depth = pd.DataFrame(df_log['CLS_DEPTH_M'])

#df_depth = df_depth.str.split(' ', expand=True)
#df_depth.columns = ['Unit', 'Depth', 'other']
#df_depth['CLS_DEPTHM'] = np.where(df_depth['Unit'].str.contains('F'), df_depth['Depth'].astype(float)/3.28084, df_depth['Depth'].astype(float))
#test = temp_df[~temp_df[0].isin(['.FT', '.M', '.F'])]


find_cols = [col for col in df_log.columns if 'TEMP' in col.upper()]
#find_cols = find_cols[0:5]+find_cols[7:9]
find_cols=find_cols[0:6]
df_log['CLS_MUDTEMP'] = coalesce(df_log, find_cols)
df_temp = pd.DataFrame(df_log['CLS_MUDTEMP'])
df_temp_split = df_temp['CLS_MUDTEMP'].str.split(' ', expand=True)
df_temp_split[0].unique()
df_temp_split.columns = ['unit', 'temp', 'no1', 'no2']
df_temp_split['CLS_MUDTEMPC'] = np.select(
    [df_temp_split['unit'].isin(['.DEGF', '.DEGFC', '.F']),
     df_temp_split['unit'].isin(['.C', '.DEGC'])],
    [pd.to_numeric(df_temp_split['temp'], errors='coerce')-32*5/9,
     pd.to_numeric(df_temp_split['temp'], errors = 'coerce')],
    default=None
    )
df_temp_split['CLS_MUDTEMPC'] = pd.to_numeric(df_temp_split['CLS_MUDTEMPC'])
df_temp = pd.DataFrame(df_temp_split['CLS_MUDTEMPC'])


find_cols = [col for col in df_log.columns if 'RESIS' in col.upper()]
remove_pats = ['GAMMA', 'SPONT', 'DENSITY', 'SONIC', 'MUD', 'TENSION', 'CALIPER', 'NEUTRON', 'CONDUCT']
find_cols = [s for s in find_cols if all(x not in s for x in remove_pats)]
df_log['CLS_RESISTIVITY'] = coalesce(df_log, find_cols)
df_log['CLS_RESISTIVITY'].describe()
df_res = pd.DataFrame(df_log['CLS_RESISTIVITY'])
df_res['CLS_RESISTIVITY'] = pd.to_numeric(df_res['CLS_RESISTIVITY'], errors='coerce')
df_res['CLS_RESISTIVITY'] = np.where(df_res['CLS_RESISTIVITY']<0.1, 0.1, df_res['CLS_RESISTIVITY'])
plt.hist(np.log10(df_res['CLS_RESISTIVITY']))



find_cols = [col for col in df_log.columns if 'PORO' in col.upper()]
remove_pats = ['GAMMA', 'SPONT',  'MUD', 'TENSION', 'CALIPER',  'CONDUCT', 'RESIS', 'FLUID', 'FACTOR']
find_cols = [s for s in find_cols if all(x not in s for x in remove_pats)]

df_log['CLS_POROSITY'] = coalesce(df_log, find_cols)
df_poro = pd.DataFrame(df_log['CLS_POROSITY'])
df_poro['CLS_POROSITY'] = pd.to_numeric(df_poro['CLS_POROSITY'] , errors='coerce')
df_poro['CLS_POROSITY'] = np.where((df_poro['CLS_POROSITY']<0) | (df_poro['CLS_POROSITY']>1),
                                   None, df_poro['CLS_POROSITY'] ) 
df_poro['CLS_POROSITY'] = pd.to_numeric(df_poro['CLS_POROSITY'] , errors='coerce')
df_poro.describe() 
df_poro['CLS_POROSITY'].isna().sum()


##------------------------------------------------------
df_output = pd.concat([df_log['UWI'],
                       df_log['read_filename'],
                       df_log['CLS_DEPTH_M'],
                       df_gamma['CLS_GAMMA'], 
                       df_temp_split['CLS_MUDTEMPC'] ,
                       df_res['CLS_RESISTIVITY'], 
                       df_poro['CLS_POROSITY']],
                      axis=1)
#df_output['Log_idx'] = df_output.groupby('')

df_output.to_csv('data/log_processed.csv')
df_output.to_parquet('data/log_processed_egb.parquet')


##----------------------------------------------------
import pandas as pd
import numpy as np
df_well =pd.read_csv('Combined3a.csv')
df_well['CLS_ELEV_M'] = np.where(df_well['fmtn']=='DVN', 
                                 pd.to_numeric(df_well['Elevation Meters_0'], errors='coerce'),
                                 pd.to_numeric(df_well['Elevation(f)_0'], errors='coerce')/3.28084)

wellcols = ['UWI', 'BHT', 'TrueTemp',  'fmtn',
            'CLS_Lat', 'CLS_Long', 'CLS_Depth_ft', 'CLS_ELEV_M', 'Krig_Temp']

df_ldvn=pd.read_parquet('data/log_processed_dvn.parquet')
df_legb=pd.read_parquet('data/log_processed_egb.parquet')

df = pd.concat([df_ldvn, df_legb])

df = df.reset_index(drop=True)
#df1 = df[df['UWI']==df['UWI'][0]]
df1 = df.copy()
#df1 = df[df['UWI']=='102112204416W500']
df1a = df1.copy()
depth_bins = [x*25 for x in range(0, 1000)]
df1a['BinDepth'] = pd.cut(df1a['CLS_DEPTH_M'], depth_bins)

df1b = df1a.groupby(['UWI', 'BinDepth']).agg({'CLS_DEPTH_M':'mean',
                                            'CLS_GAMMA':'mean',
                                           'CLS_MUDTEMPC':'mean',
                                           'CLS_RESISTIVITY':'mean',
                                           'CLS_POROSITY':'mean'}).dropna(
                                               subset=['CLS_GAMMA','CLS_MUDTEMPC','CLS_RESISTIVITY','CLS_POROSITY'], how='all').reset_index()

df1b['BinDepth'] = df1b['BinDepth'].astype(str)
df1b.to_parquet('data/log_processed_all.parquet')
                                               
df2 = df1b.merge(df_well[wellcols], how='left', left_on='UWI', right_on='UWI')
df2['CLS_BHT_DEPTH_M'] = df2['CLS_Depth_ft']/3.28084
df2['depth_diff'] = abs(df2['CLS_BHT_DEPTH_M'] - df2['CLS_DEPTH_M'])
#df2a = df2.sort_values(by=['UWI', 'depth_diff'])
df2a = df2.groupby(['UWI']).first().reset_index()

