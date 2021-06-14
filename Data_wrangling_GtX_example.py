# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import seaborn as sns
from tqdm.notebook import tqdm

Eaglebine_BHT = pd.read_excel('D:/JZP/GTX2021/Eaglebine-20210525T234229Z-001/Eaglebine/Eaglebine BHT TSC data for SPE April 21 2020.xlsx')
Eaglebine_BHT['TempC_BHT'] = (Eaglebine_BHT['BHTorMRT (maximum recorded temperature) oF'] - 32) * (5./9.)
Eaglebine_BHT.head()

print("number of unique wells: " + str(len(pd.unique(Eaglebine_BHT['UWI']))))

Eaglebine_Truth = pd.read_excel('D:/JZP/GTX2021/Eaglebine-20210525T234229Z-001/Eaglebine/Eaglebine TrueTemp_Train2.xlsx')
Eaglebine_Truth.head()

# convert to Celsius
Eaglebine_Truth['TempTrue_degC'] = (Eaglebine_Truth['True Temperature   (oF)'] - 32) * (5./9.)
print("number of unique wells in Eaglebine in training set: " + str(len(pd.unique(Eaglebine_Truth['UWI']))))

Eaglebine_Combined = Eaglebine_BHT.merge(Eaglebine_Truth, on='UWI', how='left')

# only keep from the synthetic data, the temperature at the elevation closest to the model
Eaglebine_Combined['diff_depth'] = Eaglebine_Combined['Depth sub-sea (feet)']-Eaglebine_Combined['BHT_below sea level (ft)']
Eaglebine_Combined['diff_depth_abs'] = np.abs(Eaglebine_Combined['diff_depth'])
idx = Eaglebine_Combined.groupby(['UWI'])['diff_depth_abs'].transform(min) == Eaglebine_Combined['diff_depth_abs']
TrueTempUWI = Eaglebine_Combined.loc[idx, ['UWI', 'diff_depth_abs', 'TempTrue_degC']]
TrueTempUWI = TrueTempUWI.copy(deep=True)
Eaglebine_Combined_cln = Eaglebine_BHT.merge(TrueTempUWI, on='UWI', how='left')
Eaglebine_Combined_cln.head()

len(Eaglebine_Combined_cln)

Static_log_temp = pd.read_csv('D:/JZP/GTX2021/Data_static_logs.csv')
Static_log_temp.head()

Eaglebine_Combined_cln['UWI'] = Eaglebine_Combined_cln['UWI'].astype(str)
Eaglebine_Combined_cln = Eaglebine_Combined_cln.copy(deep=True)
Eaglebine_Combined_cln['TrueTemp_datasource_syn'] = 'synthetic'
Static_log_temp['TrueTemp_datasource_stat'] = 'static_temp_logs'
Eaglebine_Combined_stat = Eaglebine_Combined_cln.merge(Static_log_temp, left_on='UWI',right_on='Well_ID', how='left')

# Coalesce columns together with priority for true temperature measurements
Eaglebine_Combined_stat['TempC_Fin'] = Eaglebine_Combined_stat['Temp (degC)'].fillna(Eaglebine_Combined_stat['TempTrue_degC'])
Eaglebine_Combined_stat['TrueTemp_datasource'] = Eaglebine_Combined_stat['TrueTemp_datasource_stat'].fillna(Eaglebine_Combined_stat['TrueTemp_datasource_syn'])
Eaglebine_Combined_stat.head()

import matplotlib.pyplot as plt
plt.close('all')
fig, ax = plt.subplots(1, 1, figsize=(4,4))

sns.scatterplot(data=Eaglebine_Combined_stat, x="TempC_BHT", y="TempC_Fin", hue='BHT_below sea level (ft)', ax=ax)

ax.set_xlim([30, 220])
ax.set_ylim([30, 220])
ax.plot([0, 220], [0, 220])
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")

Duvernay_Truth = pd.read_excel('D:/JZP/GTX2021/Duvernay-20210525T234358Z-001/Duvernay/Duvenay TrueTemp_Train.xlsx')
Duvernay_DST = pd.read_excel('D:/JZP/GTX2021/Duvernay-20210525T234358Z-001/Duvernay/Duvernay DST BHT for SPE April 20 2021.xlsx')
Duvernay_Truth.head()

# add in an extra column calculating the depth sub sea (elevation-depth)*-1 
Duvernay_DST['Depth_SS(m)']=-1*(Duvernay_DST['elevation M above sea level']-(Duvernay_DST['DST Start Depth (MD) (m)']+Duvernay_DST['DST End Depth (MD) (m)'])/2)
Duvernay_DST.head()


# merge
Duvernay_Combined = Duvernay_DST.merge(Duvernay_Truth, on='UWI', how='left')
Duvernay_Combined.head()

# keep only the synthetic model temperature data for the relevant depths for which there is BHT measurement
Duvernay_Combined['diff_depth'] = Duvernay_Combined['Depth_SS(m)']-Duvernay_Combined['Depths subsea (m)']
Duvernay_Combined['diff_depth_abs'] = np.abs(Duvernay_Combined['diff_depth'])
idx = Duvernay_Combined.groupby(['UWI'])['diff_depth_abs'].transform(min) == Duvernay_Combined['diff_depth_abs']

TrueTempUWI = Duvernay_Combined.loc[idx, ['UWI', 'diff_depth_abs', 'True Temperature (oC)']]
TrueTempUWI = TrueTempUWI.copy(deep=True)
Duvernay_Combined_cln = Duvernay_DST.merge(TrueTempUWI, on='UWI', how='left')
Duvernay_Combined_cln = Duvernay_Combined_cln.drop_duplicates(['UWI'])
Duvernay_Combined_cln.head()

len(Duvernay_Combined_cln)

Duvernay_Combined_cln['UWI'] = Duvernay_Combined_cln['UWI'].astype(str)
Duvernay_Combined_cln = Duvernay_Combined_cln.copy(deep=True)
Duvernay_Combined_cln['TrueTemp_datasource_syn'] = 'synthetic'
Static_log_temp['TrueTemp_datasource_stat'] = 'static_temp_logs'
Duvernay_Combined_stat = Duvernay_Combined_cln.merge(Static_log_temp, left_on='UWI',right_on='Well_ID', how='left')

Static_log_temp.head()

# Coalesce columns together with priority for true temperature measurements
Duvernay_Combined_stat['TempC_Fin'] = Duvernay_Combined_stat['Temp (degC)'].fillna(Duvernay_Combined_stat['True Temperature (oC)'])
Duvernay_Combined_stat['TrueTemp_datasource'] = Duvernay_Combined_stat['TrueTemp_datasource_stat'].fillna(Duvernay_Combined_stat['TrueTemp_datasource_syn'])

Duvernay_Combined_stat.head()

import matplotlib.pyplot as plt
plt.close('all')
fig, ax = plt.subplots(1, 1, figsize=(4,4))

sns.scatterplot(data=Duvernay_Combined_stat, 
                x="DST Bottom Hole Temp. (degC)",
                y="TempC_Fin",
                hue='diff_depth_abs', ax=ax)

#ax.set_xlim([30, 220])
#ax.set_ylim([30, 220])
ax.plot([0, 220], [0, 220])
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")


Duvernay = Duvernay_Combined_stat[['UWI', 'DST Bottom Hole Temp. (degC)', 'TempC_Fin','Depth_SS(m)']]
Duvernay = Duvernay.rename(columns={'DST Bottom Hole Temp. (degC)': 'BHT', 'TempC_Fin': 'TrueTemp'})
Duvernay['Field'] = 'Duvernay'

Eaglebine = Eaglebine_Combined_stat[['UWI', 'TempC_BHT', 'TempC_Fin', 'TD (ft)']]
Eaglebine = Eaglebine.rename(columns={'TempC_BHT': 'BHT', 'TempC_Fin': 'TrueTemp'})
Eaglebine['Field'] = 'Eaglebine'

combined_temperature = pd.concat((Duvernay, Eaglebine))
combined_temperature.head()

combined_temperature.to_csv('combined_temperature.csv')

plt.close('all')
fig, ax = plt.subplots(1, 1, figsize=(4,4))

sns.scatterplot(data=combined_temperature, 
                x="BHT",
                y="TrueTemp",
                hue='Field', ax=ax)

ax.plot([0, 220], [0, 220])
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")

EB_FT = pd.read_excel("D:/JZP/GTX2021/Eaglebine-20210525T234229Z-001/Eaglebine/Eaglebine formation tops SPE April 20 2021.xlsx")
display(EB_FT.head())
EB_FT.columns = [c.strip() for c in EB_FT.columns.values.tolist()]
EB_FT.rename(columns = {'UWI': 'Well_Id'}, inplace = True)

EB_MW = pd.read_excel("D:/JZP/GTX2021/Eaglebine-20210525T234229Z-001/Eaglebine/Eaglebine mud weight SPE April 21 2021.xlsx")
display(EB_MW.head())
EB_MW.columns = [c.strip() for c in EB_MW.columns.values.tolist()]
EB_MW.rename(columns = {'UWI': 'Well_Id'}, inplace = True)

EB_WH = pd.read_excel("D:/JZP/GTX2021/Eaglebine-20210525T234229Z-001/Eaglebine/Eaglebine well headers SPE April 21 2021.xlsx")
display(EB_WH.head())
EB_WH.rename(columns = {'displayapi': 'Well_Id'}, inplace = True)

EB_PS = pd.read_excel("D:/JZP/GTX2021/Eaglebine-20210525T234229Z-001/Eaglebine/SPE Eaglebine production summary April 20 2021.xlsx")
display(EB_PS.head())
EB_PS.columns = [c.strip() for c in EB_PS.columns.values.tolist()]
EB_PS.rename(columns = {'API': 'Well_Id'}, inplace = True)

DV_FT = pd.read_excel("D:/JZP/GTX2021/Duvernay-20210525T234358Z-001/Duvernay/Duvernay formation tops SPE April 20 2021.xlsx")
display(DV_FT.head())
DV_FT.columns = [c.strip() for c in DV_FT.columns.values.tolist()]
DV_FT.rename(columns = {'UWI': 'Well_Id'}, inplace = True)

DV_WH = pd.read_excel("D:/JZP/GTX2021/Duvernay-20210525T234358Z-001/Duvernay/Duvernay well headers SPE April 21 2021 .xlsx")
display(DV_WH.head())
DV_WH.columns = [c.strip() for c in DV_WH.columns.values.tolist()]
DV_WH.rename(columns = {'UWI': 'Well_Id'}, inplace = True)

DV_PS = pd.read_excel("D:/JZP/GTX2021/Duvernay-20210525T234358Z-001/Duvernay/SPE Duvernay production summary April 20 2021.xlsx")
display(DV_PS.head())
DV_PS.columns = [c.strip() for c in DV_PS.columns.values.tolist()]
DV_PS.rename(columns = {'API': 'Well_Id'}, inplace = True)

# Loop thorugh each well (Well_Id), convert each column into ndarray with Well_Id as key
# Useful functions definitions
''' list the unique well identifiers (Well_Id) in the table and the number of rows for each. The new dataframe will be used to populate the columns of the wide table'''
def wells_and_attributes(df):
    well_data = df
    columns = well_data.columns
    well_data_count = well_data['Well_Id'].value_counts()
    wells = well_data_count.index
    return well_data, wells, columns

''' 
The 'entry_to_attr' function collects the unique Well_Id and pivots entries under each Well_Id into attributes. 
The loop check for each unique well number in well_data_count, loops through columns and pivots the values in each column into a dataframe.
The name of the current column is used as prefix with an underscore "columnname_" joined with suffix as the index (from 0) of the entries for each well. 
The extracted variables are turned into dataframe and is concatenated with the existing dataframe from previous loop. 
'''
def entry_to_attr(df):
    well_data_flat = pd.DataFrame() #dataframe for storing well records after change to wide format
    well_data, wells, columns = wells_and_attributes(df) #read 'wells_and_attributes' function description
    for well in wells:
        well_data_intermediate_flat = pd.DataFrame() #temporary dataframe to store pivoted entries to new attributes for each original column
        well_data_filtered = well_data[well_data['Well_Id'] == well] #filterd for 'well' to start pivoting to new attributes
        #print(well_data_filtered)
        for column in columns: #pivots columns for each well into new attributes
            if column == 'Well_Id': #ignores the Well_Id and flattens othe columns
                continue
            else:
                att_vars = np.array(well_data_filtered[column]) #array containing entries in 'column' for 'well'
                att_count = len(att_vars) #number of entries in the array used for naming the new columns
                att_names = [column+"_"+str(x) for x in range(att_count)] #new column name with number of entries (att_count array's length) as suffix
                well_data_intermediate = pd.DataFrame(att_vars).T #intermediate flat table to be merged with the temporary dataframe
                well_data_intermediate.columns = att_names
                index = pd.Index([well], name = 'Well_Id')
                well_data_intermediate.index = index
                well_data_intermediate_flat = pd.concat([well_data_intermediate_flat, well_data_intermediate], axis = 1)
        well_data_flat = pd.concat([well_data_flat, well_data_intermediate_flat])
    return well_data_flat


'''
In this loop the entry_to_attr funciton is looped for the dataframes:
    'EB_MW', 'EB_FT', 'EB_WH', 'EB_PS', 'DV_FT', 'DV_WH', 'DV_PS'
The resulting wide-format dataframe is stored in a new datafram. 
The wide-format dataframes are concatenated into one dataframe with Well_ID as index
'''
dataframes = [EB_MW, EB_FT, EB_WH, EB_PS, DV_FT, DV_WH, DV_PS]
consolidated_well_data = pd.DataFrame()
cols = []
rows = []
# consolidated_well_data = pd.concat([consolidated_well_data, dataframes[0], dataframes[1], dataframes[2]])

files_to_upload = {'file': ['Eaglebine mud weight SPE April 21 2021.xlsx', 
                         'Eaglebine formation tops SPE April 20 2021.xlsx',
                         'Eaglebine well headers SPE April 21 2021.xlsx',
                         'SPE Eaglebine production summary April 20 2021.xlsx',
                         'Duvernay formation tops SPE April 20 2021.xlsx',
                         'Duvernay well headers SPE April 21 2021 .xlsx',
                         'SPE Duvernay production summary April 20 2021.xlsx'],
                  'df_name': ['EB_MW', 'EB_FT', 'EB_WH', 'EB_PS', 'DV_FT', 'DV_WH', 'DV_PS']}
files_to_upload['flat_df_name'] = [str(c) + '_flat' for c in files_to_upload['df_name']]
print(files_to_upload['flat_df_name'])
print(files_to_upload['df_name'][0][:2])

for i in range(len(dataframes)): #
    well_data_flat = entry_to_attr(dataframes[i])
    well_data_flat['Basin'] = files_to_upload['df_name'][i][:2]
    well_data_flat.reset_index(inplace = True)
    well_data_flat.to_excel(str(str(files_to_upload['flat_df_name'][i]) + '.' + 'xlsx'))
    print(well_data_flat.shape)
    well_data_flat.set_index(['Well_Id', 'Basin'], inplace = True)
    consolidated_well_data = pd.concat([consolidated_well_data, well_data_flat], axis = 1)
consolidated_well_data.shape
display(consolidated_well_data.head())
print('the consolidated_well_data has {} rows and {} columns'.format(consolidated_well_data.shape[0], consolidated_well_data.shape[1]))

consolidated_well_data_no_index = consolidated_well_data.reset_index()
consolidated_well_data.to_csv('consolidated_well_data.csv')
consolidated_well_data.to_excel('consolidated_well_data.xlsx')


##---------------------------------------------------------------
import os
import lasio

#Load all files at once into las and las_df to save time
folder='D:/JZP/GTX2021/well_log_files/Clean_LAS/'
all_files = os.listdir(folder)
n_files = len(all_files)

bad_files = []

las = {}
las_df = {}
mnemonics ={}
i=0
for filename in tqdm(os.listdir(folder)):
    i=i+1
    if filename.endswith(".LAS"):
        las[filename] = lasio.read(folder+'/'+filename)
        las_df[filename] = las[filename].df()
        mnemonics[filename] = las_df[filename].columns
        
        
#find out which well curves/logs are in each las file
listDF = []
for filename in tqdm(all_files):
    df = pd.DataFrame(columns = list(mnemonics[filename]), data = np.ones((1,len(mnemonics[filename]))))
    df['well_name']=filename
    listDF.append(df)
    
log_table=pd.concat(listDF)
# Here we can see which logs are in each well
log_table.head()

# see what are the most common log types
sumT = log_table.drop(columns=['well_name']).sum()
sumT.sort_values(ascending=False)

# make a table of the log types available per well
for filename in all_files:
    las_df[filename] = las_df[filename].rename_axis('DEPT').reset_index()
    

# we can extract the gamma ray values [GRWS] at 
# regular intervals to add to the dataset (every 300 ft)
WellLog = 'GRWS'

select_depth = list(np.arange(300, 30300, 300))
new_las_df = {}
extracted_df = {}
j = 0
fncnt = 0
for filename in tqdm(all_files):
    fncnt = fncnt+1

    las_df[filename] = las_df[filename].sort_values(by='DEPT')
    p = las_df[filename]
    new_las_df[filename] = p[p['DEPT'].isin(select_depth)]
    if(WellLog not in list(new_las_df[filename].columns)):
      continue
    q = new_las_df[filename][WellLog]
    Depth = new_las_df[filename]['DEPT']
    
    concat_list = list()
    column_name = list()
    for i in range(0,q.shape[0]):
        concat_list.append(q.iloc[i])
        column_name.append(str(Depth.iloc[i])+'_'+WellLog)
        
    concat_array = np.array(concat_list)
    concat_array = np.reshape(concat_array,(1,len(concat_list)))
    df = pd.DataFrame(concat_array, columns=column_name)
    df['WellName'] = filename[2:16]
    if filename[-5] == 'W':
        df['LogType'] = 'Cleaned'
    else:
        df['LogType'] = 'Raw'
    extracted_df[j] = df
    j = j+1
    
LargeDF = pd.concat(extracted_df)
LargeDF.to_csv('LogData.csv')

TemperatureData = pd.read_csv('combined_temperature.csv')
HeaderData = pd.read_csv('consolidated_well_data.csv')
LogData = pd.read_csv('LogData.csv')

Combined1 = TemperatureData.merge(HeaderData, how='left', left_on='UWI', right_on='Well_Id')
LogData['WellName']=LogData['WellName'].astype('str')
Combined1['UWI']=Combined1['UWI'].astype('str')
LogData = LogData.drop_duplicates(['WellName'])
Combined2 = Combined1.merge(LogData, how='left', left_on='UWI', right_on='WellName')
Combined2.head()

print('There are ' + str(len(Combined2)) + ' rows and ' + str(len(Combined2.columns) ) + ' columns in the dataframe')

set_split = pd.read_csv('D:/JZP/GTX2021/set_assign.csv')
set_split.head()

Combined3 = Combined2.merge(set_split, on='UWI', how='left')
Combined3.to_csv('Combined3.csv')
##-----------------
#Combined3 = pd.read_csv('Combined3.csv')

all_cols = Combined3.columns.tolist()

mis_cols = Combined3.isna().sum().reset_index()
mis_cols.to_csv('missingdata_summary.csv')
latitude_cols = [k for k in all_cols if 'LAT' in k.upper()]
longitude_cols = [k for k in all_cols if 'LONG' in k.upper()]
depth_cols = [k for k in all_cols if 
              any(k for j in ['DEPTH','TD','TVD'] if str(j) in k.upper()) & 
              ('MW' not in k.upper())]

t2 = Combined3[depth_cols]
depth_cols = ['td_0', 'Total Vertical Depth (ft)_0.1']

def coalesce(df, column_names):
    i = iter(column_names)
    column_name = next(i)
    answer=df[column_name]
    for column_name in i:
        answer = answer.fillna(df[column_name])
    return answer

depth_cols = ['TVD_DVN_MKB', 'TVD_EGB_MKB']

Combined3['TVD_EGB_MKB'] = Combined3['Total Vertical Depth (ft)_0.1']/3.28084
Combined3['TVD_DVN_MKB'] = (Combined3['Depth_SS(m)']+Combined3['Elevation Meters_0'])/3.28084

Combined3['CLS_Lat'] = coalesce(Combined3, latitude_cols)
Combined3['CLS_Long'] = coalesce(Combined3, longitude_cols)
#Combined3['CLS_Depth_ft'] = coalesce(Combined3, depth_cols)*3.28084
Combined3['CLS_Depth_ft'] = np.where(Combined3['fmtn']=='DVN',
                                     Combined3['TVD_DVN_MKB'],
                                     Combined3['TVD_EGB_MKB'])
Combined3['fmtn'] = np.where(Combined3['UWI'].str.len()==16, 'DVN', 'EGB')
#Combined3['CLS_X'] = 8541403.48 + 65434.28*Combined3['CLS_Long']
#Combined3['CLS_Y'] = 157839.73 + 108042.7*Combined3['CLS_Lat']
Combined3.to_csv('Combined3a.csv')
t2 = Combined3[Combined3['UWI']=='100142904806W500']





Combined3_dvn = Combined3[Combined3['fmtn']=='DVN']
gmin_lat = 51
gmax_lat = 57
gmin_lon = -120
gmax_lon = -110

grd_step = 100
#gridy = [x/grd_step for x in list(range(gmin_lat*grd_step,gmax_lat*grd_step))]
#gridx = [x/grd_step for x in list(range(gmin_lon*grd_step,gmax_lon*grd_step))]

gridy = list(range(gmin_lat*grd_step,gmax_lat*grd_step))
gridy = [float(i) for i in gridy]
gridx = list(range(gmin_lon*grd_step,gmax_lon*grd_step))
gridx = [float(i) for i in gridx]
# df_map = pd.DataFrame(index=np.arange(len(gridx)), columns = np.arange(len(gridy)))
# for r in range(0,df_map.shape[0]):
#     for c in range(0, df_map.shape[1]):
#         df_map.iloc[c][r] = 

df_krig = Combined3_dvn[['CLS_Long','CLS_Lat','TrueTemp']]
df_krig = df_krig.dropna()
x=np.array(df_krig['CLS_Long']*grd_step)
y=np.array(df_krig['CLS_Lat']*grd_step)
z=np.array(df_krig['TrueTemp'])
import pykrige.kriging_tools as kt
from pykrige.ok import OrdinaryKriging
import matplotlib.pyplot as plt

OK = OrdinaryKriging(
    x,
    y,
    z,
    variogram_model="linear",
    verbose=False,
    enable_plotting=False,
)

z, ss = OK.execute("grid", gridx, gridy)
df_map_dvn = pd.DataFrame(z)
df_map_dvn.columns = [int(x) for x in gridx]
df_map_dvn.index = [int(y) for y in gridy]
kt.write_asc_grid(gridx, gridy, z, filename="output.asc")
plt.imshow(z)
#plt.show()
plt.scatter(x,y)


wlist = []
for w in range(0,Combined3_dvn.shape[0]):
    wx = int(Combined3['CLS_Long'][w]*grd_step)
    wy = int(Combined3['CLS_Lat'][w]*grd_step)
    wval = df_map_dvn[wx][wy]
    wlist.append(wval)

Combined3_dvn['Krig_Temp'] = wlist    
plt.scatter(Combined3_dvn['Krig_Temp'], Combined3_dvn['TrueTemp'])
plt.plot([0, 110], [0, 110], label='1-1', c='k')
Combined3_dvn.to_csv('Combined3_dvn_krig.csv')




Combined3_egb = Combined3[Combined3['fmtn']=='EGB']
gmin_lat = 27
gmax_lat = 32
gmin_lon = -101
gmax_lon = -94

grd_step = 100
#gridy = [x/grd_step for x in list(range(gmin_lat*grd_step,gmax_lat*grd_step))]
#gridx = [x/grd_step for x in list(range(gmin_lon*grd_step,gmax_lon*grd_step))]

gridy = list(range(gmin_lat*grd_step,gmax_lat*grd_step))
gridy = [float(i) for i in gridy]
gridx = list(range(gmin_lon*grd_step,gmax_lon*grd_step))
gridx = [float(i) for i in gridx]
# df_map = pd.DataFrame(index=np.arange(len(gridx)), columns = np.arange(len(gridy)))
# for r in range(0,df_map.shape[0]):
#     for c in range(0, df_map.shape[1]):
#         df_map.iloc[c][r] = 

df_krig = Combined3_egb[['CLS_Long','CLS_Lat','TrueTemp']]
df_krig = df_krig.dropna()
x=np.array(df_krig['CLS_Long']*grd_step)
y=np.array(df_krig['CLS_Lat']*grd_step)
z=np.array(df_krig['TrueTemp'])
import pykrige.kriging_tools as kt
from pykrige.ok import OrdinaryKriging
import matplotlib.pyplot as plt

OK = OrdinaryKriging(
    x,
    y,
    z,
    variogram_model="linear",
    verbose=False,
    enable_plotting=False,
)

z, ss = OK.execute("grid", gridx, gridy)
df_map_egb = pd.DataFrame(z)
df_map_egb.columns = [int(x) for x in gridx]
df_map_egb.index = [int(y) for y in gridy]
kt.write_asc_grid(gridx, gridy, z, filename="output_egb.asc")
plt.imshow(z)
#plt.show()
plt.scatter(x,y)


wlist = []
Combined3_egb=Combined3_egb.reset_index()
for w in range(0,Combined3_egb.shape[0]):
    wx = int(Combined3_egb['CLS_Long'][w]*grd_step)
    wy = int(Combined3_egb['CLS_Lat'][w]*grd_step)
    wval = df_map_egb[wx][wy]
    wlist.append(wval)

Combined3_egb['Krig_Temp'] = wlist    
plt.scatter(Combined3_egb['Krig_Temp'], Combined3_egb['TrueTemp'])
plt.plot([80, 140], [80, 140], label='1-1', c='k')

Combined3_egb.to_csv('Combined3_egb_krig.csv')


wlist = []
for w in range(0, Combined3.shape[0]):
    wx = int(Combined3['CLS_Long'][w]*grd_step)
    wy = int(Combined3['CLS_Lat'][w]*grd_step)
    fmtn = Combined3['fmtn'][w]
    if fmtn == 'DVN':
        wval = df_map_dvn[wx][wy]
    else:
        wval = df_map_egb[wx][wy]
    wlist.append(wval)

Combined3['Krig_Temp'] = wlist
plt.scatter(Combined3['Krig_Temp'], Combined3['TrueTemp'])
plt.plot([20, 140], [20, 140], label='1-1', c='k')
Combined3.to_csv('Combined3.csv')





#Combined3=pd.read_csv('Combined3.csv')

from math import sin, cos, sqrt, atan2, radians
def latlongdistKM(lat1, long1, lat2, long2):
    R=6371
    lat1_r, lat2_r, long1_r, long2_r = (radians(x) for x in [lat1, lat2, long1, long2])
    dlon = long2_r - long1_r
    dlat = lat2_r - lat1_r
    
    a=sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2) **2
    a=abs(a)

    c=2 * atan2(sqrt(a), sqrt(1-a))
    dist = R*c
    return dist


t2 = Combined3[['UWI', 'BHT', 'TrueTemp', 'Set', 'Krig_Temp'] + Combined3.columns[Combined3.columns.str.startswith('CLS')].tolist()]
t2['key']=1
t2a = t2.copy().add_prefix('OS_')
t2j = t2.merge(t2a, how='outer', left_on = 'key', right_on='OS_key')
#t2j = t2j[0:10]
t2j['DistKM'] = t2j.apply(lambda x: latlongdistKM(x['CLS_Lat'],x['CLS_Long'],x['OS_CLS_Lat'],x['OS_CLS_Long']), axis=1)

t2j_filter = t2j.sort_values(by=['UWI','DistKM']).query('UWI != OS_UWI').groupby('UWI').head(10).reset_index().drop(['key', 'OS_key'], axis=1)

def dist_weight(x):
    
    minx = 0.1
    maxx = 0.9
    curv = 1
    startx = 10
    endx = 70
    pdc = minx + (maxx - minx) / (1+curv**(startx+endx/2-x)/endx)
    return 1-pdc
dist_weight(60)
lplot = []
for i in range(0,20):
    lplot.append(dist_weight(i))
plt.plot(lplot)

t2j_filter['dist_wt'] = dist_weight(t2j_filter['DistKM'])
t2j_filter['wtd_temp'] = t2j_filter['OS_TrueTemp'] * t2j_filter['dist_wt']
t2j_calc = t2j_filter.groupby('UWI').agg({'dist_wt':'sum','wtd_temp':'sum'}).reset_index()
t2j_calc['est_temp'] = t2j_calc['wtd_temp'] / t2j_calc['dist_wt']
t2j_calc['est_temp'] = np.where(t2j_calc['est_temp']<20,np.mean(t2j_calc['est_temp']), t2j_calc['est_temp'])

t2b = t2.merge(t2j_calc[['UWI','est_temp']], how='left', on='UWI')
plt.scatter(t2b['est_temp'], t2b['TrueTemp'])

# from sklearn.metrics import mean_squared_error
# mean_squared_error(t2b['est_temp'], t2b['TrueTemp'])

#c3save = Combined3.copy()
#Combined3=t2.copy()
t2 = t2.merge(t2j_calc[['UWI','est_temp']], how='left', on='UWI')
t2['fmtn'] = np.where(t2['UWI'].str.len()==16, 'DVN', 'EGB')
filterTraining = t2['Set']=='Training'
filterTesting = t2['Set']=='Validation_Testing'


all_train = t2[filterTraining].reset_index()
sample_rows = list(range(0,sum(filterTraining)))
import random
random.seed(10)
train_rows = random.sample(sample_rows, int(len(sample_rows)*0.7))
train_rows.sort()
train = all_train[all_train.index.isin(train_rows)]
test = all_train[~all_train.index.isin(train_rows)]
validation_testing = t2[filterTesting]

labelcol = 'TrueTemp'
feature_cols = ['BHT','CLS_Lat', 'CLS_Long', 'CLS_Depth_ft', 'est_temp', 'Krig_Temp','fmtn']
feature_cols = ['BHT', 'CLS_Lat', 'CLS_Long', 'Krig_Temp']
#feature_cols = ['BHT','CLS_Lat', 'CLS_Long', 'est_temp', 'fmtn']
import matplotlib.pyplot as plt
import seaborn as sns

plt.hist(train[labelcol])

sns.pairplot(train, y_vars = labelcol, x_vars = train[[labelcol]+feature_cols].columns.values)
g=sns.PairGrid(train)
g.map(sns.scatterplot)


from pycaret.regression import *
clf1 = setup(data=train[[labelcol]+feature_cols], target=labelcol, html=False, feature_selection=True)
cm = compare_models(n_select = 5)
best_model = tune_model(cm[0])

plot_model(best_model, plot='feature')
plot_model(best_model, plot='residuals')

from sklearn.inspection import plot_partial_dependence
plot_partial_dependence(best_model, X=train[[labelcol]+feature_cols], features=feature_cols[:-1] )


best_model = finalize_model(best_model)


preds = predict_model(best_model, data = test)


ax = plt.gca()
ax.scatter(preds[labelcol], preds['Label'], color='red')
#ax.scatter(preds['temp_grad'], preds['depth_subsea'], color='green')
#ax.set_yscale('log')
#ax.set_xscale('log')

val_df = predict_model(best_model, data=validation_testing).reset_index()
val_output = val_df[['UWI','Label']]
val_output.columns=['UWI', 'TrueTemp']
val_output.to_csv('predictions.csv')

import zipfile
zipfile.ZipFile('predictions.zip', mode='w').write("predictions.csv")


##-----------------


filterTraining = Combined3['Set']=='Training'
Combined3[filterTraining].to_csv('training.csv', index=False)

filterTesting = Combined3['Set']=='Validation_Testing'
Combined3[filterTesting].to_csv('val_data_no_label.csv', index=False)

train = pd.read_csv('training.csv')
train.head()

from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(train['BHT'].values.reshape(-1, 1),
                             train['TrueTemp'].values.reshape(-1, 1))
x = np.arange(200)
y_hat = reg.predict(x.reshape(-1, 1))



fig, ax = plt.subplots(1, 1, figsize=(4,4))

sns.scatterplot(data=train, x="BHT", y="TrueTemp", hue='Field', ax=ax)

ax.set_xlim([30, 220])
ax.set_ylim([30, 220])
ax.plot([0, 220], [0, 220], label='1-1', c='k')

ax.plot(x, y_hat, label='linear pred', c='r')

plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")

test_data = pd.read_csv('val_data_no_label.csv')
test_data.head()


prediction = reg.predict(test_data['BHT'].values.reshape(-1, 1))
test_data['TrueTemp']=prediction
test_data[['UWI','TrueTemp']].to_csv('predictions.csv')

import zipfile
zipfile.ZipFile('predictions.zip', mode='w').write("predictions.csv")
