library(dplyr)
library(data.table)
df = read.csv('data/Final_Preds.csv')
nnet_res = read.csv('data/Krig_NNET_Error.csv')
df =  left_join(df, nnet_res, by='UWI')
temp_df = df %>%
  filter(Lat >= 54 & Lat <= 55 & Long >= -116 & Long <= -115) %>%
  select(UWI, Lat, Long, EX_TrueTemp, Krig_EX_TrueTemp ,NNET_Pred_TempC, Set)

rad = function(x){
  x = x * pi / 180
}

latlong_dist = function(lat1, long1, lat2, long2){
  R=6371
  lat1_r = rad(lat1)
  lat2_r = rad(lat2)
  long1_r = rad(long1)
  long2_r = rad(long2)

  dlon = long2_r - long1_r
  dlat = lat2_r - lat1_r
  
  a=sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2) **2
  a=abs(a)
  
  c=2 * atan2(sqrt(a), sqrt(1-a))
  dist = R*c
  return(dist)
}

latlong_dist(54.42636, -115.4056, 54.80172, -115.1147)

df_well = df %>%
  mutate(DST_TVDMKB = DST_TVDMSS + ELEV_M) %>%
  select(UWI, Lat, Long,DST_TVDMSS ,TVDMSS,EX_TrueTemp, NNET_Pred_TempC, Krig_NNET_Error) %>% mutate(key = 1) 

df_well2 = df_well %>%
  rename_at(vars(names(.)), function(x)paste0(x,'_OS')) %>%
  full_join(df_well, by=c('key_OS' = 'key')) %>%
  select(!ends_with('_OS'), everything()) %>%
  select(-contains('key')) %>%
  mutate(DistKM = mapply(latlong_dist, Lat, Long, Lat_OS, Long_OS)) 


df_well2a = df_well2 %>%
  filter(UWI != UWI_OS) %>%
  arrange(UWI, DistKM) %>%
  group_by(UWI) %>%
  slice(1)

df_well2a = df_well2a %>%
  mutate(Dist3D = ((DistKM*1000)^2 + (DST_TVDMSS_OS-DST_TVDMSS)^2)^0.5 ) %>%
  mutate(EX_TrueTempGrad = EX_TrueTemp / DST_TVDMSS,
         EX_TrueTempGrad_OS =  EX_TrueTemp_OS/ DST_TVDMSS_OS,
         NNET_TempGrad = NNET_Pred_TempC / DST_TVDMSS,
         NNET_Pred_TempGrad_OS = NNET_Pred_TempC_OS / DST_TVDMSS_OS)

# df_well2a$DisCo_NNTempGrad = ifelse(df_well2a$DistKM<3 & !is.na(df_well2a$EX_TrueTemp_OS) & df_well2a$DST_TVDMKB_OS>0 & df_well2a$DST_TVDMKB_OS<5000,
#                                 df_well2a$NNET_Pred_TempC/df_well2a$DST_TVDMKB*0.5 + df_well2a$EX_TrueTemp_OS/df_well2a$DST_TVDMKB_OS*0.5,
#                                 df_well2a$NNET_Pred_TempC/df_well2a$DST_TVDMKB)

# df_well2a$DisCo_NNTemp = ifelse(df_well2a$DistKM<5 & !is.na(df_well2a$EX_TrueTemp_OS) , 
#                                     df_well2a$NNET_Pred_TempC*0.0 + df_well2a$EX_TrueTemp_OS*1.0,
#                                     df_well2a$NNET_Pred_TempC)

dist_weight = function(x){
  minx = 0.4
  maxx = 0.99
  curv = 500
  startx = 500
  endx = 5000
  pdc = minx + (maxx - minx) / (1+curv^((startx+endx/2-x)/endx))
  return(1-pdc)
}
  


df_well2a$Dist3DWt = mapply(dist_weight, df_well2a$Dist3D)
plot(df_well2a$Dist3D, df_well2a$Dist3DWt)
df_well2a$DWT_NNET_Temp = df_well2a$Dist3DWt*df_well2a$NNET_Pred_TempC_OS+(1-df_well2a$Dist3DWt)*df_well2a$NNET_Pred_TempC
mean(abs(df_well2a$DWT_NNET_Temp-df_well2a$EX_TrueTemp), na.rm=T)
sd(abs(df_well2a$DWT_NNET_Temp-df_well2a$EX_TrueTemp), na.rm=T)
plot(df_well2a$DWT_NNET_Temp, df_well2a$EX_TrueTemp)

names(df_well2a)
library(leaps)
labelcol='EX_TrueTemp'
feature_cols = c('Lat', 'Long', 'DST_TVDMSS', 'TVDMSS', 'NNET_Pred_TempC', 'Krig_NNET_Error',  'Lat_OS', 'Long_OS', 'DST_TVDMSS_OS',
                 'TVDMSS_OS', 'EX_TrueTemp_OS', 'NNET_Pred_TempC_OS', 'Krig_NNET_Error_OS', 'DistKM', 'Dist3D', 'EX_TrueTempGrad_OS',
                 'NNET_TempGrad')
df_mod = df_well2a[,c(labelcol, feature_cols)]
#df_mod = na.omit(df_mod)
leap1 = regsubsets(EX_TrueTemp~., data = df_mod, method='exhaustive')
plot(leap1, scale='bic')
summary.leapsc2 <- summary(leap1)
leaptabc2<-data.frame(summary.leapsc2$which[which.min(summary.leapsc2$bic),])
leaptabc2 <- add_rownames(leaptabc2, "Variable")
colnames(leaptabc2) <- c("Variable","InOut")
leaptabc2 <- leaptabc2%>%filter(InOut==TRUE)
leaptabc2 <- leaptabc2[2:nrow(leaptabc2),]
LMfmla <- paste(leaptabc2$Variable, collapse = "+")
LMfmla <- paste('EX_TrueTemp', "~", LMfmla)
lmModel <- lm(LMfmla, data=df_mod) 
summary(lmModel)
plot(predict(lmModel, newdata = df_mod), df_mod$EX_TrueTemp)
mean(abs(predict(lmModel, newdata = df_mod)- df_mod$EX_TrueTemp), na.rm=T)
sd(abs(predict(lmModel, newdata = df_mod)- df_mod$EX_TrueTemp), na.rm=T)
plot(df_mod$NNET_Pred_TempC, df_mod$EX_TrueTemp)
mean(abs(df_mod$NNET_Pred_TempC- df_mod$EX_TrueTemp), na.rm=T)
sd(abs(df_mod$NNET_Pred_TempC- df_mod$EX_TrueTemp), na.rm=T)

df2 = cbind(df, df_mod[,names(df_mod)[!(names(df_mod) %in% names(df))]]) %>%
  mutate(NewPreds = predict(lmModel, newdata = .))

df2_val = df2 %>% filter(Set == 'Validation_Testing') %>% select(UWI, NewPreds) %>% rename(TrueTemp = NewPreds)
fwrite(df2_val, 'data/Rpreds.csv')




df_well2a$DisCo_NNTemp = ifelse(df_well2a$DisCo_NNTempGrad<0.04, df_well2a$DisCo_NNTempGrad  * df_well2a$DST_TVDMKB, df_well2a$NNET_Pred_TempC)

plot(df_well2a$DisCo_NNTemp, df_well2a$EX_TrueTemp)
plot(df_well2a$NNET_Pred_TempC, df_well2a$EX_TrueTemp)
df_val = df_well2a %>% select(UWI, DisCo_NNTemp) %>% rename(TrueTemp=DisCo_NNTemp)
fwrite(df_val, 'data/Rpreds.csv')



library(leaps)
labelcol = 'EX_TrueTemp'
orig_feature_cols = c('Krig_NNET_Error','NNET_Pred_TempC','Lat','Long','DST_TempGrad_CpM','DST_TVDMSS','TVDMSS',
                 'DST_TempC','Krig_CLS_GAMMA','Krig_CLS_MUDTEMPC','Krig_CLS_RESISTIVITY',
                 'Krig_CLS_POROSITY','Krig_CLS_TrueTempGrad_CpM','Krig_EX_TrueTemp',
                 'Krig_EX_TrueTempGrad_CpM','Krig_CLS_CumOil_bbl','Krig_CLS_CumGas_mcf',
                 'Krig_CLS_CumWater_bbl','Krig_CLS_OGR_bblmmcf','DVN_Cluster.0',
                 'DVN_Cluster.1','DVN_Cluster.2','DVN_Cluster.3','EGB_Cluster.0',
                 'EGB_Cluster.1','EGB_Cluster.2','EGB_Cluster.3')
df_mod = select(df, labelcol, all_of(orig_feature_cols))

leap1 = regsubsets(EX_TrueTemp~., data = df_mod, method='exhaustive')
plot(leap1, scale='adjr2')
summary.leapsc2 <- summary(leap1)
leaptabc2<-data.frame(summary.leapsc2$which[which.max(summary.leapsc2$adjr2),])
leaptabc2 <- add_rownames(leaptabc2, "Variable")
colnames(leaptabc2) <- c("Variable","InOut")
leaptabc2 <- leaptabc2%>%filter(InOut==TRUE)
leaptabc2 <- leaptabc2[2:nrow(leaptabc2),]
LMfmla <- paste(leaptabc2$Variable, collapse = "+")
LMfmla <- paste('EX_TrueTemp', "~", LMfmla)
lmModel <- lm(LMfmla, data=df_mod) 
summary(lmModel)

df_mod$NewPreds = predict(lmModel, newdata = df_mod)
plot(df_mod$NewPreds, df_mod$EX_TrueTemp)
plot(df_mod$NNET_Pred_TempC, df_mod$EX_TrueTemp)

mean(abs(df_mod$NNET_Pred_TempC-df_mod$EX_TrueTemp), na.rm=T)
mean(abs(df_mod$NewPreds - df_mod$EX_TrueTemp), na.rm=T)

df$NewPreds = predict(lmModel, newdata = df)
df_out = df %>% filter(Set == 'Validation_Testing') %>% select(UWI, NewPreds) %>% rename(TrueTemp=NewPreds)
fwrite(df_out,  'data/Rpreds.csv')


df_mod2 = na.omit(df_mod)

library(xgboost)
feature_cols = c('NNET_Pred_TempC', 'Lat', 'Long', 'TVDMSS')
feature_cols = leaptabc2$Variable
feature_cols = c('Krig_NNET_Error','NNET_Pred_TempC', 'DVN_Cluster.0',  'DVN_Cluster.2',
                 'EGB_Cluster.3', 'DST_TVDMSS',  'Lat', 'Long')
smp_size = floor(0.8 * nrow(df_mod2))
set.seed(1)
train_ind = sample(seq_len(nrow(df_mod2)), size = smp_size)
train = df_mod2[train_ind,]
test = df_mod2[-train_ind,]

xginput = as.matrix( train[,feature_cols])

xgb1 <- xgboost(data = xginput, 
                label = train[,labelcol], 
                eta = 0.01,
                gamma = 2,
                max_depth = 10, 
                subsample = 0.9,
                colsample_bytree = 0.9,
                # seed = 1,
                nrounds = 10000,
                eval_metric = "rmse",
                objective = "reg:squarederror",
                nthread = 1
)

plot(predict(xgb1, newdata = as.matrix(test[,feature_cols])), test$EX_TrueTemp)
lines(c(0,200), c(0,200), col='red')
mean(abs(predict(xgb1, newdata = as.matrix(test[,feature_cols]))- test$EX_TrueTemp))
sd(abs(predict(xgb1, newdata = as.matrix(test[,feature_cols]))- test$EX_TrueTemp))
plot(test$NNET_Pred_TempC, test$EX_TrueTemp)
lines(c(0,200), c(0,200), col='red')
mean(abs(test$NNET_Pred_TempC - test$EX_TrueTemp), na.rm=T)
sd(abs(test$NNET_Pred_TempC - test$EX_TrueTemp), na.rm=T)

df$NewPreds = predict(xgb1, newdata = as.matrix(df[,feature_cols]))
df_out = df %>% filter(Set == 'Validation_Testing') %>% select(UWI, NewPreds) %>% rename(TrueTemp=NewPreds)
fwrite(df_out,  'data/Rpreds.csv')



library(ggplot2)
df$NNerror = abs(df$EX_TrueTemp-df$NNET_Pred_TempC)
ggplot(df[df$fmtn=='DVN',], aes(x=Long, y=Lat, color = NNerror)) + 
  geom_point() + 
  scale_color_gradient2(midpoint = mean(df$NNerror, na.rm=T),  low='blue', mid='white', high='red', space='Lab')

ggplot(df[df$fmtn=='EGB',], aes(x=Long, y=Lat, color = Cluster, size = NNerror)) + 
  geom_point(shape=1) 

ggplot(df[df$fmtn=='DVN',], aes(x=Long, y=Lat, color = Cluster, size = NNerror)) + 
  geom_point(shape=1) 


higherr_uwi_dvn =df %>% filter(fmtn=='DVN') %>% filter(abs(NNerror)>10) %>% arrange(NNerror)
table(higherr_uwi_dvn$Cluster)

ggplot(higherr_uwi_dvn, aes(x=Long, y=Lat, color = NNerror)) + 
  geom_point() + 
  scale_color_gradient2(midpoint = mean(df$NNerror, na.rm=T),  low='red', mid='green',  high='blue')

ggplot(df[df$fmtn=='EGB',], aes(x=Cluster, y=log10(NNerror))) + geom_boxplot()


bad_clus = c('EGB_Cluster 3', 'DVN_Cluster 0', 'DVN_Cluster 2')


bad_clus_df = df[df$Cluster %in% bad_clus,]
good_df = df[!(df$Cluster %in%  bad_clus), ]

df_mod2b = select(bad_clus_df, labelcol, all_of(orig_feature_cols), Krig_NNET_Error)

leap1a = regsubsets(EX_TrueTemp~., data = df_mod2b, method='exhaustive')
plot(leap1, scale='adjr2')
summary.leapsc2 <- summary(leap1)
leaptabc2<-data.frame(summary.leapsc2$which[which.max(summary.leapsc2$adjr2),])
leaptabc2 <- add_rownames(leaptabc2, "Variable")
colnames(leaptabc2) <- c("Variable","InOut")
leaptabc2 <- leaptabc2%>%filter(InOut==TRUE)
leaptabc2 <- leaptabc2[2:nrow(leaptabc2),]
LMfmla <- paste(unique(c(leaptabc2$Variable, gsub(' ', '.', bad_clus))), collapse = "+")
LMfmla <- paste('EX_TrueTemp', "~", LMfmla)
lmModel <- lm(LMfmla, data=df_mod2b) 
lmModel = lm(EX_TrueTemp ~ NNET_Pred_TempC+Krig_CLS_GAMMA+Krig_CLS_RESISTIVITY+
               Krig_CLS_TrueTempGrad_CpM+Krig_EX_TrueTemp+DVN_Cluster.0+
               EGB_Cluster.3+Krig_NNET_Error,
             data = df_mod2b)
summary(lmModel)


bad_clus_df$NewPreds = predict(lmModel, newdata = bad_clus_df)
plot(bad_clus_df$NewPreds, bad_clus_df$EX_TrueTemp)
plot(bad_clus_df$NNET_Pred_TempC, bad_clus_df$EX_TrueTemp)

print('nnet')
mean(abs(bad_clus_df$NNET_Pred_TempC-bad_clus_df$EX_TrueTemp), na.rm=T)
sd(abs(bad_clus_df$NNET_Pred_TempC-bad_clus_df$EX_TrueTemp), na.rm=T)
print('newpreds')
mean(abs(bad_clus_df$NewPreds - bad_clus_df$EX_TrueTemp), na.rm=T)
sd(abs(bad_clus_df$NewPreds - bad_clus_df$EX_TrueTemp), na.rm=T)

good_df$NewPreds = good_df$NNET_Pred_TempC

df = rbind(good_df, bad_clus_df)
plot(df$NNET_Pred_TempC, df$EX_TrueTemp)
plot(df$NewPreds, df$EX_TrueTemp)
mean(abs(df$NewPreds - df$EX_TrueTemp), na.rm=T)
mean(abs(df$NNET_Pred_TempC-df$EX_TrueTemp), na.rm=T)
df_out = df %>% filter(Set == 'Validation_Testing') %>% select(UWI, NewPreds) %>% rename(TrueTemp=NewPreds)

fwrite(df_out,  'data/Rpreds.csv')




