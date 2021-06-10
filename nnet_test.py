# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 21:21:19 2021

@author: JZ2018
"""

import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import keras
from keras.callbacks import  EarlyStopping
from  keras.layers import Dropout

import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/mod_data_clust.csv')

kvars = ['CLS_GAMMA', 'CLS_MUDTEMPC', 'CLS_RESISTIVITY', 'CLS_POROSITY', 
         'CLS_TrueTempGrad_CpM', 'EX_TrueTemp', 'EX_TrueTempGrad_CpM',
         'CLS_CumOil_bbl', 'CLS_CumGas_mcf', 'CLS_CumWater_bbl', 'CLS_OGR_bblmmcf']

df['Cluster'] = df['fmtn'] + '_' + df['Cluster']
cluster_dummies = pd.get_dummies(df['Cluster'])
df = pd.concat([df, cluster_dummies], axis=1)

clust_cols = cluster_dummies.columns.tolist()

labelcol = 'EX_TrueTemp'
feature_cols = [  'Lat', 'Long', 'DST_TempGrad_CpM',
                'DST_TVDMSS','DST_TempC'] + ['Krig_'+ x for x in kvars] + clust_cols
                

all_train = df[df['Set']=='Training'].reset_index()
all_train_save = all_train.copy()
validation_test = df[df['Set']=='Validation_Testing'].reset_index()

all_train = all_train_save[all_train_save['fmtn']=='DVN'].reset_index(drop=True)

sample_rows = list(range(0,all_train.shape[0]))
import random
random.seed(123)
train_rows = random.sample(sample_rows, int(len(sample_rows)*0.8))
train_rows.sort()
train = all_train[all_train.index.isin(train_rows)]
test = all_train[~all_train.index.isin(train_rows)]

train = train[[labelcol]+ feature_cols]

X  = train[feature_cols]
Y = train[labelcol]




def nnet_compile(add_layers,last_layer_node,def_alpha,drop=0):
    model= Sequential()
    model.add(Dense(X.shape[1], input_dim = X.shape[1], kernel_initializer = 'normal', activation  = keras.layers.LeakyReLU(alpha=def_alpha)))
    
    i=0
    while i < add_layers:
        print(f'add layer {i}')
        model.add(Dense(X.shape[1], kernel_initializer = 'normal', activation  = keras.layers.LeakyReLU(alpha=def_alpha)))
        model.add(Dropout(drop))
        i=i+1
        
    #model.add(Dense(12, kernel_initializer = 'normal', activation  = keras.layers.LeakyReLU(alpha=def_alpha)))
    model.add(Dense(last_layer_node, kernel_initializer = 'normal', activation  = keras.layers.LeakyReLU(alpha=def_alpha)))
    #model.add(Dense(4, kernel_initializer = 'normal', activation  = keras.layers.LeakyReLU(alpha=def_alpha)))
    model.add(Dense(1, kernel_initializer = 'normal'))
    model.compile(loss='mean_absolute_error', optimizer='adam')
    print('model compiled')
    return model


#model= Sequential()
# model.add(Dense(X.shape[1], input_dim = X.shape[1], kernel_initializer = 'normal',  activation = 'relu'))
#model.add(Dense(X.shape[1], input_dim = X.shape[1], kernel_initializer = 'normal', activation  = keras.layers.LeakyReLU(alpha=0.3)))
# model.add(Dense(X.shape[1], kernel_initializer = 'normal', activation  = keras.layers.LeakyReLU(alpha=0.3)))
# model.add(Dense(17, kernel_initializer = 'normal', activation = 'relu'))


#model.add(Dense(1, kernel_initializer = 'normal'))
#model.compile(loss='mean_absolute_error', optimizer='adam')

model = nnet_compile(add_layers = 1, last_layer_node=8, def_alpha = 0.1, drop=0)

history = model.fit(X, Y, epochs = 2000, batch_size=20, validation_split=0.2)
model.evaluate(X, Y)

# es= EarlyStopping(monitor='val_loss',mode='min', verbose = 1, patience = 100)
# history = model.fit(X, Y, epochs = 5000, batch_size=10, validation_split=0.2, callbacks=[es])

plt.plot(np.log10(history.history['loss']),label='train')
plt.plot(np.log10(history.history['val_loss']),label='test')
plt.legend()
plt.show()

test_X= test[feature_cols]
preds_train = model.predict(X)
preds = model.predict(test_X)

plt.scatter(preds_train, train[labelcol], label= 'Train')
plt.scatter(preds, test[labelcol], label='Test')
plt.title('Neural Network DL')
plt.xlabel('Temperature C')
plt.ylabel('Temperature C')
plt.plot(plt.xlim(), plt.ylim(), linestyle='--', color='k')
plt.legend()

test['preds'] = preds
test['error'] = test[labelcol]-test['preds']
print(np.mean(abs(test['error'])))



epoch_ranges = [1000,2000,4000]
batch_ranges = [2,10,20,40]
layer_ranges = [1,5]
alpha_ranges = [0.05, 0.1, 0.3]
midnode_ranges = [4,8,12]
drop_ranges = [0,0.2]

import itertools
sens_df = pd.DataFrame(list(itertools.product(
    *[epoch_ranges, batch_ranges, layer_ranges, alpha_ranges, midnode_ranges, drop_ranges])),
    columns = ['epoch', 'batch', 'layer', 'alpha', 'midnode', 'drop'])

sens_df_out = sens_df.copy()
sens_df_out['test_mae'] =  0.00

#for i in range(0, sens_df.shape[0]):
for i in range(0, sens_df.shape[0]):
    df_row = sens_df.iloc[i]
    model = nnet_compile(add_layers = int(df_row['layer']), 
                         last_layer_node=int(df_row['midnode']), 
                         def_alpha = df_row['alpha'], 
                         drop=df_row['drop'])
    history = model.fit(X, Y, epochs = int(df_row['epoch']), batch_size=int(df_row['batch']), validation_split=0.1)
    #preds_train = model.predict(X)
    preds = model.predict(test_X)
    test['preds'] = preds
    test['error'] = test[labelcol]-test['preds']
    mae_test = np.mean(abs(test['error']))
    sens_df_out['test_mae'][i] = mae_test
    del model
    del history
    
    

val_pred = model.predict(validation_test[feature_cols])
validation_test['Label']= val_pred
val_output = validation_test[['UWI','Label']]
val_output.columns=['UWI', 'TrueTemp']
val_output.to_csv('predictions.csv')

import zipfile
zipfile.ZipFile('predictions.zip', mode='w').write("predictions.csv")
