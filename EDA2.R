library(data.table)
library(dplyr)
df = fread('data/mod_data.csv')
labelcol = 'EX_TrueTemp'
featurecols = c('TVDMSS', 'Lat', 'Long', 'DST_TempGrad_CpM',
                'DST_TVDMSS','DST_TempC',"Krig_CLS_GAMMA","Krig_CLS_MUDTEMPC", "Krig_CLS_RESISTIVITY" ,
                "Krig_CLS_POROSITY" ,"Krig_CLS_TrueTempGrad_CpM", "Krig_EX_TrueTemp"  ,"Krig_EX_TrueTempGrad_CpM" , "Krig_CLS_CumOil_bbl",
                "Krig_CLS_CumGas_mcf", "Krig_CLS_CumWater_bbl", "Krig_CLS_OGR_bblmmcf")

df_mod = df %>%
  filter(fmtn == 'EGB') %>%
  select(all_of(c(labelcol, featurecols))) %>% na.omit()
library(leaps)

leap1 = regsubsets(EX_TrueTemp~., data = df_mod, method='exhaustive')
plot(leap1, scale='adjr2')

#selectcols = c('Lat', 'Long', 'DST_TVDMSS', 'Krig_CLS_RESISTIVITY', 'Krig_EX_TrueTemp', 'Krig_EX_TrueTempGrad_CpM', 'Krig_CLS_CumGas_mcf', 'Krig_CLS_CumWater_bbl')
#selectcols = c('Lat', 'Long', 'DST_TVDMSS',  'Krig_CLS_RESISTIVITY', 'Krig_EX_TrueTemp', 'Krig_EX_TrueTempGrad_CpM', 'Krig_CLS_CumWater_bbl', 'Krig_CLS_CumGas_mcf')

##-----------------------------

feature_df = select(df_mod, all_of(featurecols))
PreddataInputsq <- feature_df
for( k in names(feature_df)){
  PreddataInputsq$newvar <- PreddataInputsq[[k]]^2
  #PreddataInputsq <- cbind(PreddataInputsq, PreddataInputsq$newvar)
  names(PreddataInputsq)[names(PreddataInputsq) == 'newvar'] <- paste(k, ".", k, sep="")
}


v2 <- feature_df
v3 <- v2
for( m in names(v3)){		
  for( n in names(v3)){
    if(m == n){next}
    v2$newvar <- v2[[paste(n)]] / v2[[paste(m)]]
    names(v2)[names(v2) == 'newvar'] <- paste(n, ".div.", m, sep="")
  }
  
}
v2 <- v2[, (ncol(v3)+1):ncol(v2)]
v2 <- as.data.frame(v2)



datacols <- ncol(feature_df)+1
feature_df2 <- model.matrix(~.^2, data=feature_df)
feature_df2 <- as.data.frame(feature_df2)
feature_df2 <- feature_df2[,datacols:ncol(feature_df2)]	

feature_df3 <- cbind(PreddataInputsq, feature_df2, v2)
feature_df3 <- as.data.frame(feature_df3)

names(feature_df3) <- make.names(names(feature_df3), unique=TRUE)

##-----------------------------

#df_mod2 = select(df, all_of(c(labelcol, selectcols)))
df_mod2 = cbind(df_mod$EX_TrueTemp, feature_df3)
names(df_mod2)[1] = 'EX_TrueTemp'

leap1 = regsubsets(EX_TrueTemp~., data = df_mod2, method='seqrep')
#plot(leap1, scale='adjr2')
summary.leapsc2 <- summary(leap1)
leaptabc2<-data.frame(summary.leapsc2$which[which.max(summary.leapsc2$adjr2),])
leaptabc2 <- add_rownames(leaptabc2, "Variable")
colnames(leaptabc2) <- c("Variable","InOut")
leaptabc2 <- leaptabc2%>%filter(InOut==TRUE)
leaptabc2 <- leaptabc2[2:nrow(leaptabc2),]
LMfmla <- paste(leaptabc2$Variable, collapse = "+")
LMfmla <- paste('EX_TrueTemp', "~", LMfmla)
lmModel <- lm(LMfmla, data=df_mod2) 
summary(lmModel)

df_mod2$PredLabel = predict(lmModel, newdata = df_mod2)
df_mod2a = cbind(select(df, UWI, Set), df_mod2)
plot(df_mod2a$PredLabel,df_mod2a$EX_TrueTemp)

df_val = df_mod2a %>% filter(Set == 'Validation_Testing') %>% select(UWI, PredLabel) %>% rename(TrueTemp=PredLabel)
fwrite(df_val, 'data/Rpreds.csv')

##----------------------



lmod1 = lm(EX_TrueTemp~., data = df_mod2)
summary(leap1)
summary(lmod1)
df$PredLabel = predict(lmod1, newdata = df)
plot(df$PredLabel,df$EX_TrueTemp)
df_val = df %>% filter(Set == 'Validation_Testing') %>% select(UWI, PredLabel) %>% rename(TrueTemp=PredLabel)
fwrite(df_val, 'data/Rpreds.csv')
