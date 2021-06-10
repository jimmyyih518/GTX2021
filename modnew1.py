# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 00:04:49 2021

@author: JZ2018
"""

import pandas as pd
import numpy as np
from helper_funs import *
import matplotlib.pyplot as plt
import seaborn as sns

##----------------------------------------
kvars = ['CLS_GAMMA', 'CLS_MUDTEMPC', 'CLS_RESISTIVITY', 'CLS_POROSITY', 
         'CLS_TrueTempGrad_CpM', 'EX_TrueTemp', 'EX_TrueTempGrad_CpM',
         'CLS_CumOil_bbl', 'CLS_CumGas_mcf', 'CLS_CumWater_bbl', 'CLS_OGR_bblmmcf']

 
for k in range(0, len(kvars)):
    print(k)
    Krig_var = kvars[k]
    temp_df = pd.read_csv(f'data/Krig_{Krig_var}.csv')
    temp_df = temp_df.iloc[:, 1:]
    if k == 0:
        out_df = temp_df.copy()
    else:
        out_df = out_df.merge(temp_df, how='left', on = 'UWI')
        
df = pd.read_csv('data/proc_log_well_combine.csv')
df_join = df.merge(out_df, how='left', on='UWI')
df_join['TVDMSS'] = np.where(df_join['TVDMSS'].isna(), df_join['DST_TVDMSS'], df_join['TVDMSS'])
df_join['CLS_TrueTemp'] = coalesce(df_join, ['SLOG_TEMPC', 'TrueTemp'])
df_join['log_CLS_TrueTempGrad_CpM'] = np.log10(df_join['CLS_TrueTempGrad_CpM'])
df_join['log_TVDMSS'] = np.log10(df_join['TVDMSS'])

##----------------------------------------
Combined3 = pd.read_csv('Combined3.csv')
ex_df = Combined3[['UWI', 'TrueTemp']]
ex_df.columns = ['UWI', 'EX_TrueTemp']
df_join = df_join.merge(ex_df, how='left', on='UWI')
sns.scatterplot(df_join['CLS_TrueTemp'], df_join['EX_TrueTemp'], hue=df_join['fmtn'])

df_join.to_csv('data/mod_data.csv')
##----------------------------------------
#df_join = pd.read_csv('data/mod_data.csv')
df_join = pd.read_csv('data/mod_data_clust.csv')
df_join['Cluster'] = df_join['fmtn'] + '_' + df_join['Cluster']

df_join['DST_TempGrad_CpM'] = np.where(df_join['DST_TempGrad_CpM']<0.025,None, df_join['DST_TempGrad_CpM'])
df_join['DST_TempGrad_CpM'] = np.where(df_join['DST_TempGrad_CpM']>0.1,None, df_join['DST_TempGrad_CpM'])
df_join['DST_TempGrad_CpM'] = df_join['DST_TempGrad_CpM'].fillna(df_join['DST_TempGrad_CpM'].mean())


#labelcol = 'CLS_TrueTemp'
#labelcol = 'CLS_TrueTempGrad_CpM'
labelcol = 'EX_TrueTemp'
feature_cols = [  'Lat', 'Long', 'DST_TempGrad_CpM',
                'DST_TVDMSS','DST_TempC', 'Cluster'] + ['Krig_'+ x for x in kvars]

# feature_cols = ['DST_TempC', 'DST_TVDMSS',
#                 'Krig_CLS_POROSITY', 'Krig_EX_TrueTempGrad_CpM',
#                 'Long', 'Krig_CLS_CumGas_mcf']

# #lr dvn
# feature_cols = ['DST_TempGrad_CpM','Krig_CLS_RESISTIVITY' ,'Krig_CLS_POROSITY', 'Krig_CLS_TrueTempGrad_CpM','Krig_EX_TrueTemp',
#                 'Krig_CLS_OGR_bblmmcf', 'Krig_CLS_CumGas_mcf']

# #lr egb
# feature_cols = ['Long', 'DST_TempGrad_CpM', 'DST_TVDMSS', 
#                 'DST_TempC', 'Krig_CLS_RESISTIVITY', 
#                 'Krig_EX_TrueTemp', 'Krig_CLS_CumGas_mcf']


#sns.scatterplot(np.log10(df_join['DST_TempGrad_CpM']), np.log10(df_join[labelcol]), hue=df_join['fmtn'])

df_mod = df_join[[labelcol]+feature_cols+['UWI','Set', 'fmtn', 'TVDMSS']]

all_train = df_mod[df_mod['Set']=='Training'].reset_index()
all_train_save = all_train.copy()
validation_test = df_mod[df_mod['Set']=='Validation_Testing'].reset_index()
# set_assign = pd.read_csv('data/set_assign.csv')
# set_assign2 = set_assign.merge(df_join, how='left', on='UWI')
##----------------------------------------

all_train = all_train_save[all_train_save['fmtn']=='DVN'].reset_index(drop=True)

##----------------------------------------------
from pycaret.clustering import *

clf_clus = setup(data = all_train[feature_cols], normalize = True, session_id = 123)
km1 =  create_model('kmeans')
#print(km1)
#kmodes= create_model('kmeans', num_clusters = 4)
#print(kmodes)

kpreds = assign_model(km1)
sns.scatterplot(kpreds['Long'], kpreds['Lat'], hue = kpreds['Cluster'])
plt.title('Duvernay Data Clusters')
plot_model(km1, plot='elbow')
plot_model(km1, plot='silhouette')

all_train = predict_model(km1, all_train)
df_dvn = df_join[df_join['fmtn']=='EGB']
df_dvn = predict_model(km1, df_dvn)
df_dvn.to_csv('data/clust_egb.csv')
#feature_cols.append('Cluster')

##----------------------------------------------
df_dvn = pd.read_csv('data/clust_dvn.csv')
df_egb = pd.read_csv('data/clust_egb.csv')

df_join = pd.concat([df_dvn, df_egb])
df_join.to_csv('data/mod_data_clust.csv')
##----------------------------------------
sample_rows = list(range(0,all_train.shape[0]))
import random
random.seed(254)
train_rows = random.sample(sample_rows, int(len(sample_rows)*0.8))
train_rows.sort()
train = all_train[all_train.index.isin(train_rows)]
test = all_train[~all_train.index.isin(train_rows)]

##----------------------------------------

#dvn filter https://static.ags.aer.ca/files/document/OFR/OFR_2017_02.pdf
# train = train[(train[labelcol]/train['TVDMSS']*1000>=22) & 
#               (train[labelcol]/train['TVDMSS']*1000<=38)]
#train = train[(train['index']!=531) & (train['index']!=719 )]
train[labelcol].describe()
#plt.hist(train[labelcol]/train['TVDMSS']*1000)
plt.hist(train[labelcol])

#egb filter
# train['TempGradCalc'] = train[labelcol]/train['TVDMSS']*1000
# train['TempGradCalc'].hist()
# train = train[(train[labelcol]/train['TVDMSS']*1000>=20) & 
#               (train[labelcol]/train['TVDMSS']*1000<=60)]
#sns.pairplot(train, y_vars = labelcol, x_vars = train[[labelcol]+feature_cols].columns.values)
#sns.scatterplot(train['TVDMSS'], train[labelcol])
#sns.scatterplot(train['Krig_CLS_TrueTempGrad_CpM'], train[labelcol], hue=train['fmtn'])
# g=sns.PairGrid(train)
# g.map(sns.scatterplot)

#sns.scatterplot(all_train['Long'], all_train['Lat'], hue=all_train[labelcol])
#sns.scatterplot(all_train['TVDMSS'], all_train[labelcol], hue=all_train['Krig_CLS_CumGas_mcf'])
##----------------------------------------

from pycaret.regression import *
# feature_cols = ['Krig_CLS_GAMMA', 'Krig_CLS_MUDTEMPC', 'Krig_CLS_RESISTIVITY',
#                 'Krig_CLS_POROSITY', 'Krig_CLS_TrueTempGrad_CpM', 'TVDMSS', 
#                 'Long', 'DST_TVDMSS','DST_TempC']
#feature_cols.remove('Krig_CLS_MUDTEMPC')
#feature_cols.remove('DST_TempGrad_CpM')
#feature_cols.remove('DST_TVDMSS')
clf1 = setup(data=train[[labelcol]+feature_cols], target=labelcol, html=False, feature_selection=True)
# clf1 = setup(data=train[[labelcol]+feature_cols], target=labelcol, html=False, feature_selection=True,
#              normalize_method='robust', transformation = True, pca=True, pca_components = 6, remove_outliers=True,
#              create_clusters=True, polynomial_features = True, feature_interaction = True, profile = True)


cm = compare_models(n_select = 12)
best_model = tune_model(cm[0])
#best_model = cm[0]
plot_model(best_model, plot='feature')
plot_model(best_model, plot='residuals')
plot_model(best_model, plot='rfe')
plot_model(best_model, plot='vc')
plot_model(best_model, plot='learning')
plot_model(best_model, plot='cooks')
plot_model(best_model, plot='error')

mod_df_clf = clf1[6]
sns.scatterplot(mod_df_clf['Long'], mod_df_clf['Lat'], hue = mod_df_clf['Cluster'])
mod_features=mod_df_clf.columns.tolist()
from sklearn.inspection import plot_partial_dependence
pdp_plot = plot_partial_dependence(best_model, X=train[mod_features], features=mod_features )
pdp_plot.figure_.suptitle('Partial Dependence of Model Features')
pdp_plot.figure_.subplots_adjust(hspace=1)

best_model = finalize_model(best_model)

preds = predict_model(best_model, data = test)
ax = plt.gca()
ax.scatter(preds[labelcol], preds['Label'], color='red')
preds['error'] = preds['Label'] -preds[labelcol]
np.mean(abs(preds['error']))




clf2 = setup(data=all_train[[labelcol]+feature_cols], target=labelcol, html=False, feature_selection=True)

# clf2 = setup(data=all_train[[labelcol]+feature_cols], target=labelcol, html=False)

# best_model2 = create_model('lar')

cm = compare_models(n_select = 10)
best_model = tune_model(cm[0])

preds  = predict_model(best_model, data = df_join)

ax = plt.gca()
ax.scatter(preds[labelcol], preds['Label'], color='red')
sns.scatterplot(preds[labelcol], preds['Label'], hue = preds['Long'])
preds['error'] = preds['Label'] - preds[labelcol]

df_filt_preds = preds[preds['fmtn']=='DVN']
sns.scatterplot(df_filt_preds['Long'], df_filt_preds['Lat'], hue=abs(df_filt_preds['error']))
plt.title('Residual Map')


val_df = predict_model(best_model, data=validation_test).reset_index()
val_output = val_df[['UWI','Label']]
val_output.columns=['UWI', 'TrueTemp']
#val_output.to_csv('predictions.csv')
#val_output.to_csv('predictions_dvn.csv')
#val_output.to_csv('predictions_egb.csv')

##----------------------------------------
import shap
explainer = shap.TreeExplainer(best_model)
X_train = all_train[feature_cols]
shap_values = explainer.shap_values(X_train)
shap.summary_plot(shap_values, features=X_train, feature_names=X_train.columns, show=False)
plt.savefig('shapely_plot.png')
plt.close()

##----------------------------------------

dvn_val = pd.read_csv('predictions_dvn.csv')
egb_val = pd.read_csv('predictions_egb.csv')

val_combine = pd.concat([dvn_val[dvn_val['UWI'].str.len()==16], egb_val[egb_val['UWI'].str.len()==14]])
val_combine = val_combine.iloc[:, 1:]
val_combine.to_csv('predictions.csv')
#val_combine = pd.read_csv('data/Rpreds.csv')

import zipfile
zipfile.ZipFile('predictions.zip', mode='w').write("predictions.csv")

