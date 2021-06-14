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

from keras.layers import  *
import tensorflow

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

##-----------------------------------------------------------------
clust_cols = ['DVN_Cluster 0', 'DVN_Cluster 1', 'DVN_Cluster 2',
       'DVN_Cluster 3', 'EGB_Cluster 0', 'EGB_Cluster 1', 'EGB_Cluster 2',
       'EGB_Cluster 3']

df = pd.read_csv('data/Final_Preds.csv')
NNET_krigRes = pd.read_csv('data/Krig_NNET_Error.csv')
df = df.merge(NNET_krigRes, how='left', on='UWI')

labelcol = 'EX_TrueTemp'
feature_cols = ['Krig_NNET_Error', 'NNET_Pred_TempC' ,'Lat', 'Long', 'DST_TempGrad_CpM',
                'DST_TVDMSS','DST_TempC'] + ['Krig_'+ x for x in kvars] + clust_cols
                

all_train = df[df['Set']=='Training'].reset_index()
all_train_save = all_train.copy()
validation_test = df[df['Set']=='Validation_Testing'].reset_index()

#all_train = all_train_save[all_train_save['fmtn']=='DVN'].reset_index(drop=True)

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


def nnet_compile(add_layers,layer_nodes,def_alpha,drop=0):
    model= Sequential()
    #model.add(Dense(X.shape[1], input_dim = X.shape[1], kernel_initializer = 'normal', activation  = keras.layers.LeakyReLU(alpha=def_alpha)))
    model.add(Dense(X.shape[1], input_dim = X.shape[1], kernel_initializer = 'normal'))
    model.add(tensorflow.keras.layers.LeakyReLU(alpha=0.2))
    i=0
    while i < add_layers:
        print(f'add layer {i} with {layer_nodes[i]} nodes')
        model.add(Dense(layer_nodes[i], kernel_initializer = 'normal', activation  = keras.layers.LeakyReLU(alpha=def_alpha)))
        model.add(Dropout(drop))
        i=i+1
        
    #model.add(Dense(12, kernel_initializer = 'normal', activation  = keras.layers.LeakyReLU(alpha=def_alpha)))
    # model.add(Dense(layer_nodes[i], kernel_initializer = 'normal', activation  = keras.layers.LeakyReLU(alpha=def_alpha)))
    # print(f'add final layer with {layer_nodes[i] nodes')
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

model = nnet_compile(add_layers = 3, layer_nodes=[24, 12, 6], def_alpha = 0.3, drop=0)
#es= EarlyStopping(monitor='loss',mode='min', verbose = 1, baseline = 4, patience = 1000)
# from keras.callbacks import ModelCheckpoint

# mc = ModelCheckpoint('best_model.h5', monitor = 'loss', mode='min', save_best_only = True)

history = model.fit(X, Y, epochs = 5000, batch_size=10)

# from keras.models import load_model
# saved_model = load_model('best_model.h5')

model.evaluate(X, Y)
model.evaluate(test[feature_cols], test[labelcol])

# es= EarlyStopping(monitor='val_loss',mode='min', verbose = 1, patience = 100)
# history = model.fit(X, Y, epochs = 5000, batch_size=10, validation_split=0.2, callbacks=[es])
# plt.plot(history.history['loss'],label='train')

# loscount = history.history['loss']
# loscount = pd.DataFrame(loscount)
# loscount.to_csv('data/nn_loss.csv')
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



# epoch_ranges = [1000,2000,4000]
# batch_ranges = [2,10,20,40]
# layer_ranges = [1,5]
# alpha_ranges = [0.05, 0.1, 0.3]
# midnode_ranges = [4,8,12]
# drop_ranges = [0,0.2]


epoch_ranges = [5000,8000]
batch_ranges = [5]
layer_ranges = [1]
alpha_ranges = [0.3]
midnode_ranges = [20,22,24]
drop_ranges = [0]

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
    #history = model.fit(X, Y, epochs = int(df_row['epoch']), batch_size=int(df_row['batch']), validation_split=0.1)
    history = model.fit(X, Y, epochs = int(df_row['epoch']), batch_size=int(df_row['batch']))
    #preds_train = model.predict(X)
    preds = model.predict(test_X)
    test['preds'] = preds
    test['error'] = test[labelcol]-test['preds']
    mae_test = np.mean(abs(test['error']))
    sens_df_out['test_mae'][i] = mae_test
    del model
    del history
    
#sens_df_out.to_csv('data/nn_sens_df.csv')    

#sens_df_plot = pd.read_csv('data/nn_sens_df.csv')
sens_df_plot = pd.concat([sens_df_plot, sens_df_out])
sens_df_plot.to_csv('data/nn_sens_df.csv')
sens_df_plot = sens_df_out[sens_df_out['test_mae']>0]
sns.boxplot(x=sens_df_plot['batch'], y=sens_df_plot['test_mae'])
#less layer, higher alpha, more midnode, no drop, smaller batch


##-----------------------------------------------------------------
model = nnet_compile(add_layers = 2, layer_nodes=[12,5], def_alpha = 0.3, drop=0)
history2 = model.fit(all_train[feature_cols], all_train[labelcol], epochs = 20000, batch_size=5, validation_split=0.1)
plt.plot(np.log10(history2.history['loss']),label='train')
plt.plot(np.log10(history2.history['val_loss']),label='test')
# plt.plot(history2.history['loss'],label='train')
# plt.plot(history2.history['val_loss'],label='test')
plt.legend()
plt.show()
model.evaluate(all_train[feature_cols], all_train[labelcol])

model.save('data/final_nnet.h5')


##------------------------------------------------------------------
new_features = ['NNET_Pred_TempC','Krig_NNET_Error',
                'EGB_Cluster 3', 'DVN_Cluster 0', 'DVN_Cluster 2']
new_features = feature_cols
mod1a = Sequential()
#mod1.add(Dense(X.shape[1]))
mod1a.add(Dense(len(new_features), kernel_initializer = 'normal'))
mod1a.add(tensorflow.keras.layers.LeakyReLU(alpha=0.3))
mod1a.add(Dense(int(len(new_features)/2), kernel_initializer = 'normal'))
mod1a.add(tensorflow.keras.layers.LeakyReLU(alpha=0.3))
mod1a.add(Dense(int(len(new_features)/2/2), kernel_initializer = 'normal'))
mod1a.add(tensorflow.keras.layers.LeakyReLU(alpha=0.3))
mod1a.add(Dense(1, kernel_initializer = 'normal'))
mod1a.compile(loss='mean_absolute_error', optimizer='adam')

# history_mod1 = mod1.fit(all_train[feature_cols], all_train[labelcol], epochs = 2000, batch_size=5, validation_split=0.1)
# plt.plot(np.log10(history_mod1.history['loss']),label='train')
# plt.plot(np.log10(history_mod1.history['val_loss']),label='test')
# mod1.evaluate(all_train[feature_cols], all_train[labelcol])
# mod1.save('data/final_nnet_2.h5')
# mod1.add(tensorflow.keras.layers.LeakyReLU(alpha=0.2))

#mod1a = mod1
# from keras.models import load_model
# mod1a = load_model('data/final_nnet_2.h5')
#preds1 = mod1.predict(all_train[feature_cols])
#preds1a = mod1a.predict(all_train[feature_cols])
#mod1b = tensorflow.keras.models.load_model('data/final_nnet.h5', custom_objects={'LeakyReLu': tensorflow.nn.leaky_relu})

history_1a = mod1a.fit(all_train[new_features], all_train[labelcol], epochs = 5000, batch_size=600, validation_split=0.1)
plt.plot(np.log10(np.log10(history_1a.history['loss'])),label='train')
plt.plot(np.log10(np.log10(history_1a.history['val_loss'])),label='test')
plt.xscale('log')
# plt.plot(history2.history['loss'],label='train')
# plt.plot(history2.history['val_loss'],label='test')
plt.legend()
plt.show()
mod1a.evaluate(all_train[new_features], all_train[labelcol])
mod1a.save('data/final_nnet_2.h5')

preds = mod1a.predict(all_train[new_features])
sns.scatterplot(pd.DataFrame(preds)[0], all_train[labelcol], hue=all_train['fmtn'])


# mod1 = load_model('data/final_nnet.h5', custom_objects={'leaky_relu':tensorflow.nn.leaky_relu})
# mod1 = load_model('data/final_nnet.h5')
# import tensorflow as tf
# tf.keras.models.load_model('data/final_nnet.h5', custom_objects={'LeakyReLu': tf.nn.leaky_relu})

df_nn_preds = pd.read_csv('data/Final_Preds.csv')
#df_nn_preds['NN_tempgrad'] = df_nn_preds['NNET_Pred_TempC'] / 
df_nn_preds['NewPreds'] = np.where(df_nn_preds['NNET_Pred_TempC']<df_nn_preds['Krig_EX_TrueTemp'], df_nn_preds['Krig_EX_TrueTemp'],df_nn_preds['NNET_Pred_TempC'])
val_output = df_nn_preds[df_nn_preds['Set'] == 'Validation_Testing']
val_output = val_output[['UWI', 'NewPreds']]
val_output = val_output[['UWI', 'NNET_Pred_TempC']]
val_output.columns = ['UWI', 'TrueTemp']
val_output['TrueTemp'].hist()
val_output.to_csv('predictions.csv')

df_nn_preds_filt = df_nn_preds[df_nn_preds['fmtn']=='DVN']
sns.scatterplot(df_nn_preds_filt['Long'], df_nn_preds_filt['Lat'], hue=df_nn_preds_filt['NNET_Pred_TempC'])


val_pred = mod1a.predict(validation_test[new_features])
validation_test['Label']= val_pred
val_output = validation_test[['UWI','Label']]
val_output.columns=['UWI', 'TrueTemp']
val_output.to_csv('predictions.csv')

import zipfile
zipfile.ZipFile('predictions.zip', mode='w').write("predictions.csv")


# outpreds = model.predict(df[feature_cols])
# df['NNET_Pred_TempC']  = outpreds
# sns.scatterplot(df['NNET_Pred_TempC'], df['EX_TrueTemp'], hue=df['fmtn'])
# df.to_csv('data/Final_Preds.csv')
