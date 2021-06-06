# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 18:05:40 2021

@author: JZ2018
"""
import os
import pandas as pd
import numpy as np
import re
import gc 



las_seps = ['~Well', '~Curve', '~Params', '~Other', '~ASCII']

def white_rm(x):
    output = re.sub(' +', ' ', x)
    return output

def log_parse_one(fname):
    
    df = pd.read_csv(fname, sep='\n', header=None)
    log_dict = {'~Well':None, '~Curve':None, '~Params':None, '~ASCII':None}
    for s in range(0, len(las_seps)):
        print(las_seps[s])
        info_start = df[df[0].str.contains(las_seps[s])].index[0]+1
        if las_seps[s]=='~Other':
            continue
        elif las_seps[s] == '~ASCII':
            info_end = df.shape[0]
            temp_df = df[info_start:info_end]
            temp_df[0] = temp_df[0].map(white_rm)
            temp_df = temp_df[0].str.split(' ', expand=True)
            log_dict[las_seps[s]] = pd.concat([log_dict[las_seps[s]], temp_df])
    
        elif las_seps[s]=='~Well' or las_seps[s]=='~Params':
            info_end = df[df[0].str.contains(las_seps[s+1])].index[0]
            
            temp_df = df[info_start:info_end]
            if temp_df.shape[0]>0:
                temp_df[0] = temp_df[0].map(white_rm)
                temp_df = temp_df[0].str.split(':', expand=True)
                temp_df[[2,3]] = temp_df[0].str.split(' ', 1, expand=True)
                temp_df = pd.DataFrame(temp_df)
                temp_df[4] = temp_df[1] + '_' + temp_df[2]
                outcols = temp_df[1]
                temp_df = temp_df[[3]].transpose()
                temp_df.columns = outcols
            else:
                temp_df = pd.DataFrame()
        elif las_seps[s]=='~Curve':
            info_end = df[df[0].str.contains(las_seps[s+1])].index[0]
            temp_df = df[info_start:info_end]
            if temp_df.shape[0]>0:
                temp_df[0] = temp_df[0].map(white_rm)
                temp_df = temp_df[0].str.split(':', expand=True)
                temp_df[2] = temp_df[0]+temp_df[1]
                outcols = temp_df[2].tolist()
                temp_df = temp_df[[2]].transpose()
                temp_df.columns = outcols
            else:
                temp_df = pd.DataFrame()
        log_dict[las_seps[s]] = pd.concat([log_dict[las_seps[s]], temp_df])
    
    return log_dict

#log1 = log_parse_one(fname)
def log_dict2df(log_dict):
   
    df_ascii = log_dict['~ASCII']
    #nunique = df_ascii.apply(pd.Series.nunique)
    #drop_cols = nunique[nunique==1].index
    df_ascii = df_ascii.drop(0, axis=1)
    df_ascii.columns = log_dict['~Curve'].columns
    
    df_params = log_dict['~Params']
    df_params=df_params.loc[df_params.index.repeat(df_ascii.shape[0])]
    
    df_well = log_dict['~Well']
    df_well=df_well.loc[df_well.index.repeat(df_ascii.shape[0])]
    
    df_out = pd.concat([df_ascii.reset_index(drop=True), df_params.reset_index(drop=True)], axis=1)
    df_out = pd.concat([df_out.reset_index(drop=True), df_well.reset_index(drop=True)], axis=1)
    
    return df_out

#df_c = log_dict2df(log1)

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



fdir = 'data/well_log_files/Clean_LAS/'
flist = os.listdir(fdir)
fname = flist[0]

startrange = [x*100+1024 for x in list(range(0,4))]
endrange = startrange[1:]+[len(flist)]

meter_defs = ['METER', 'M']
feet_defs = ['FT', 'FEET', 'F']

for i in range(0, len(startrange)):
    print(startrange[i])
    log_df = pd.DataFrame()
    for f in range(startrange[i], endrange[i]):
    #for f in range(0, len(flist)):    
        fname=flist[f]
        print(f)
        #print(f/200)
        fpath = fdir+fname
        temp_log = log_parse_one(fpath)
        temp_df = log_dict2df(temp_log)
        temp_df.columns = [' '.join(s.split()) for s in temp_df.columns]
        print(temp_df.columns[0])
        
        if any(s in temp_df.columns[0].upper() for s in feet_defs):
            temp_df.iloc[:,0] = temp_df.iloc[:,0].apply(lambda x: pd.to_numeric(x, errors='coerce') / 3.28084)
        else:
            temp_df.iloc[:,0] = temp_df.iloc[:,0].apply(lambda x: pd.to_numeric(x, errors='coerce'))
        temp_df.rename(columns={temp_df.columns[0]:'CLS_DEPTH_M'}, inplace=True)
        temp_df['read_filename'] = fname
        temp_df = temp_df.rename(columns=renamer())
        log_df = pd.concat([log_df.reset_index(drop=True), temp_df.reset_index(drop=True)])
        #del temp_df
        #gc.collect()
        
    #log_df.to_csv('data/combine_log.csv')    
    print(log_df.shape)
    log_df.to_parquet(f'data/egbtemplogs/temp_combine_log{i}.parquet')        
    del log_df   
    gc.collect()
    
    
    
    
    
    
    
    
    