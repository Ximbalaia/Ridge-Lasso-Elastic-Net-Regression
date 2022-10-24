library(tidyverse)
library(corrplot)
library(plyr)
library(gridExtra)
library(caret)
library(mltools)
library(ggrepel)
library(car)
library(glmnet)
library(caTools)
library(e1071)

#Although the regression was made on my own, most of the pre processing data and EDA was based on this work: 
#<https://www.kaggle.com/code/erikbruin/house-prices-lasso-xgboost-and-a-detailed-eda>  

#This file is slightly different from the Markdown.

#*Dataset:*<https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data>  

train = read.csv('train.csv')
test = read.csv('test.csv')
tt = read.csv('train.csv')

test$Id = NULL
train$Id = NULL

test$SalePrice = NA
train_x = rbind(train, test)

ggplot(train_x[!is.na(train_x$SalePrice), ], aes(SalePrice)) +
  geom_histogram(aes(y = ..density..), bins = 150, colour = 'cadetblue1', fill = 'skyblue') + 
  geom_density(lwd = 1.2, linetype = 1, colour = 2) + 
  scale_x_continuous(breaks = seq(0, 800000, by = 100000))

summary(train_x$SalePrice)

numericVars = which(sapply(train_x, is.numeric))
numericVarNames = names(numericVars)

train_x_numVars = train_x[, numericVars]
cor_numVars = cor(train_x_numVars, use = 'pairwise.complete.obs') #correlations of all numeric variables
#sort on decreasing correlations with SalePrice
cor_sorted = as.matrix(sort(cor_numVars[, 'SalePrice'], decreasing = T))
#select only high correlations
CorHigh = names(which(apply(cor_sorted, 1, function(x) abs(x) > 0.5)))
cor_numVars = cor_numVars[CorHigh, CorHigh]
corrplot.mixed(cor_numVars, tl.col = 'black', tl.pos = 'lt')

##Overall quality variable
ggplot(train_x[!is.na(train_x$SalePrice), ], aes(x = factor(OverallQual), y = SalePrice)) + 
  geom_boxplot(col = 'red') + labs(x = 'Overall Quality') + 
  scale_y_continuous(breaks = seq(0, 800000, by = 100000))

##Above Grade Living Area
ggplot(train_x, aes(x = GrLivArea, y = SalePrice)) +
  geom_point(col = 'blue') + geom_smooth(method = 'lm', se = F, color = 'black', aes(group = 1)) + 
  scale_y_continuous(breaks = seq(0, 800000, by = 100000))

#Missing Values
NAcol = colnames(train_x)[colSums(is.na(train_x)) > 0]
sort(colSums(sapply(train_x[NAcol], is.na)), decreasing = T)

#Fixing it
#PoolQC
train_x$PoolQC[is.na(train_x$PoolQC)] = 'No Pool'

#MiscFeature
train_x$MiscFeature[is.na(train_x$MiscFeature)] = 'None'

#Alley
train_x$Alley[is.na(train_x$Alley)] = 'No Alley'

#Fence
train_x$Fence[is.na(train_x$Fence)] = 'No Fence'

#FireplaceQu
train_x$FireplaceQu[is.na(train_x$FireplaceQu)] = 'No Fireplace'

#LotFrontage
train_x$LotFrontage[is.na(train_x$LotFrontage)] = median(train_x$LotFrontage, na.rm = T)

#GarageType, GarageFinish, GarageQual, GarageCond
train_x = train_x %>%
  mutate_at(c('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'), ~replace_na(., 'No Garage'))

#GarageYrBlt
train_x$GarageYrBlt[is.na(train_x$GarageYrBlt)] = train_x$YearBuilt[is.na(train_x$GarageYrBlt)]
train_x[train_x$GarageYrBlt == 2207, ]$train_x = 2007

#BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2
train_x = train_x %>%
  mutate_at(c('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'), ~replace_na(., 'No Basement'))

#MasVnrType
train_x$MasVnrType[is.na(train_x$MasVnrType)] = 'None'

#MasVnrArea
train_x$MasVnrArea[is.na(train_x$MasVnrArea)] = median(train_x$MasVnrArea, na.rm = T)

#Electrical
train_x$Electrical[is.na(train_x$Electrical)] = 'SBrkr'

#MSZoning
train_x$MSZoning[is.na(train_x$MSZoning)] = median(train_x$MSZoning, na.rm = T)

#BsmtFullBath
train_x$BsmtFullBath[is.na(train_x$BsmtFullBath)] = median(train_x$BsmtFullBath, na.rm = T)

#BsmtHalfBath
train_x$BsmtHalfBath[is.na(train_x$BsmtHalfBath)] = median(train_x$BsmtHalfBath, na.rm = T)

#Functional
train_x$Functional[is.na(train_x$Functional)] = median(train_x$Functional, na.rm = T)

#Exterior1st
train_x$Exterior1st[is.na(train_x$Exterior1st)] = names(sort(-table(train_x$Exterior1st))) [1]

#Exterior2nd
train_x$Exterior2nd[is.na(train_x$Exterior2nd)] = names(sort(-table(train_x$Exterior2nd))) [1]

#BsmtFinSF1
train_x$BsmtFinSF1[is.na(train_x$BsmtFinSF1)] = 0

#BsmtFinSF2
train_x$BsmtFinSF2[is.na(train_x$BsmtFinSF2)] = 0

#BsmtUnfSF
train_x$BsmtUnfSF[is.na(train_x$BsmtUnfSF)] = 0

#TotalBsmtSF
train_x$TotalBsmtSF[is.na(train_x$TotalBsmtSF)] = 0

#KitchenQual
train_x$KitchenQual[is.na(train_x$KitchenQual)] = 'TA' #replace with most common value

#GarageCars
train_x$GarageCars[is.na(train_x$GarageCars)] = 0

#GarageArea
train_x$GarageArea[is.na(train_x$GarageArea)] = 0

#SaleType
train_x$SaleType[is.na(train_x$SaleType)] = names(sort(-table(train_x$SaleType)))[1]

#check NA values again
colnames(train_x)[colSums(is.na(train_x)) > 0]

## Drop Utilities
train_x$Utilities = NULL

## Create house variables
train_x$Remod = as.factor(ifelse(train_x$YearBuilt == train_x$YearRemodAdd, 0, 1)) #0 = No remodeling, 1 = Remodeling
train_x$IsNew <- as.factor(ifelse(train_x$YrSold == train_x$YearBuilt, 1, 0)) #0 = No, 1 = Yes


###########################################################################################################
## MAKE CHARACTER VARIABLES FACTOR
str(train_x)
## MSSubClass
train_x$MSSubClass = factor(train_x$MSSubClass, levels = c('20', '30', '40', '45', '50', '60', '70', '75', '80', '85', '90',
                                                           '120', '150', '160', '180', '190'),
                            labels = c(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15))

##MSZoning
train_x$MSZoning = factor(train_x$MSZoning, levels = c('C (all)', 'RM', 'RL', 'RH', 'FV'),
                          labels = c(0, 1, 2, 3, 4))

##Street
train_x$Street = factor(train_x$Street, levels = c('Pave', 'Grvl'), labels = c(0, 1))

##Alley
train_x$Alley = factor(train_x$Alley, levels = c('No Alley', 'Pave', 'Grvl'), labels =c(0, 1, 2))

##LotShape
train_x$LotShape = factor(train_x$LotShape, levels = c('IR3', 'IR2', 'IR1', 'Reg'), labels = c(0, 1, 2, 3))

##LandContour
train_x$LandContour = factor(train_x$LandContour, levels = c('Low', 'HLS', 'Bnk', 'Lvl'),
                             labels = c(0, 1, 2, 3))

##LotConfig
train_x$LotConfig = factor(train_x$LotConfig, levels = c('FR3', 'FR2', 'CulDSac', 'Corner', 'Inside'),
                           labels = c(0, 1, 2, 3, 4))

##LandSlope
train_x$LandSlope = factor(train_x$LandSlope, levels = c('Sev', 'Mod', 'Gtl'),
                           labels = c(0, 1, 2))

##Neighborhood
#Check where is the better Neighborhood
nb1 <- ggplot(train_x, aes(x=reorder(Neighborhood, SalePrice, FUN=median), y=SalePrice)) +
  geom_bar(stat='summary', fun = "median", fill='blue') + labs(x='Neighborhood', y='Median SalePrice') +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_y_continuous(breaks= seq(0, 800000, by=50000)) +
  geom_label(stat = "count", aes(label = ..count.., y = ..count..), size=3) +
  geom_hline(yintercept=163000, linetype="dashed", color = "red") #dashed line is median SalePrice

nb2 <- ggplot(train_x, aes(x=reorder(Neighborhood, SalePrice, FUN=mean), y=SalePrice)) +
  geom_bar(stat='summary', fun = "mean", fill='blue') + labs(x='Neighborhood', y="Mean SalePrice") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_y_continuous(breaks= seq(0, 800000, by=50000)) +
  geom_label(stat = "count", aes(label = ..count.., y = ..count..), size=3) +
  geom_hline(yintercept=163000, linetype="dashed", color = "red") #dashed line is median SalePrice

grid.arrange(nb1, nb2)

##Create NeighRich
train_x$NeighRich[train_x$Neighborhood %in% c('StoneBr', 'NridgHt', 'NoRidge')] = 2
train_x$NeighRich[!train_x$Neighborhood %in% c('MeadowV', 'IDOTRR', 'BrDale', 'StoneBr', 'NridgHt', 'NoRidge')] = 1
train_x$NeighRich[train_x$Neighborhood %in% c('MeadowV', 'IDOTRR', 'BrDale')] = 0
train_x$NeighRich = as.factor(train_x$NeighRich)

#train_x$Neighborhood = factor(train_x$Neighborhood, levels = c('Veenker', 'Timber', 'StoneBr', 'Somerst', 'SawyerW', 'Sawyer',
#                                                               'SWISU', 'OldTown', 'NWAmes', 'NridgHt', 'NPkVill', 'NoRidge',
#                                                               'NAmes', 'Mitchel', 'MeadowV', 'IDOTRR', 'Gilbert', 'Edwards',
#                                                               'Crawfor', 'CollgCr', 'ClearCr', 'BrkSide', 'BrDale', 'Blueste', 'Blmngtn' ),
#                              labels = c(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24))

##Remove Neighborhood
train_x$Neighborhood = NULL

##Condition1
train_x$Condition1 = factor(train_x$Condition1, levels = c('RRAe', 'RRNe', 'PosA', 'PosN', 'RRAn', 'RRNn', 'Norm', 'Feedr', 'Artery'),
                            labels = c(0, 1, 2, 3, 4, 5, 6, 7, 8))

##Condition2
train_x$Condition2 = factor(train_x$Condition2, levels = c('RRAe', 'RRNe', 'PosA', 'PosN', 'RRAn', 'RRNn', 'Norm', 'Feedr', 'Artery'),
                            labels = c(0, 1, 2, 3, 4, 5, 6, 7, 8))

##BldgType
train_x$BldgType = factor(train_x$BldgType, levels = c('Twnhs', 'TwnhsE', 'Duplex', '2fmCon', '1Fam'),
                          labels = c(0, 1, 2, 3, 4))

##HouseStyle
train_x$HouseStyle = factor(train_x$HouseStyle, levels = c('SLvl', 'SFoyer', '2.5Unf', '2.5Fin', '2Story', '1.5Unf', '1.5Fin', '1Story'),
                            labels = c(0, 1, 2, 3, 4, 5, 6, 7))

##RoofStyle
train_x$RoofStyle = factor(train_x$RoofStyle, levels = c('Shed', 'Mansard', 'Hip', 'Gambrel', 'Gable', 'Flat'),
                           labels = c(0, 1, 2, 3, 4, 5))

##RoofMatl
train_x$RoofMatl = factor(train_x$RoofMatl, levels = c('WdShngl', 'WdShake', 'Tar&Grv', 'Roll', 'Metal', 'Membran', 'CompShg', 'ClyTile'),
                          labels = c(0, 1, 2, 3, 4, 5, 6, 7))

##Exterior1st
train_x$Exterior1st = factor(train_x$Exterior1st, levels = c('WdShing', 'Wd Sdng', 'VinylSd', 'Stucco', 'Stone', 'PreCast',
                                                             'Plywood', 'Other', 'MetalSd', 'ImStucc', 'HdBoard', 'CemntBd',
                                                             'CBlock', 'BrkFace', 'BrkComm', 'AsphShn', 'AsbShng'),
                             labels = c(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16))	
	
##Exterior2nd
train_x$Exterior2nd = factor(train_x$Exterior2nd, levels = c('Wd Shng', 'Wd Sdng', 'VinylSd', 'Stucco', 'Stone', 'PreCast',
                                                             'Plywood', 'Other', 'MetalSd', 'ImStucc', 'HdBoard', 'CmentBd',
                                                             'CBlock', 'BrkFace', 'Brk Cmn', 'AsphShn', 'AsbShng'),
                             labels = c(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16))		

##MasVnrType
train_x$MasVnrType = factor(train_x$MasVnrType, levels = c('Stone', 'None', 'CBlock', 'BrkFace', 'BrkCmn'),
                            labels = c(0, 1, 2, 3, 4))

##ExterQual
train_x$ExterQual = factor(train_x$ExterQual, levels = c('Po', 'Fa', 'TA', 'Gd', 'Ex'),
                           labels = c(0, 1, 2, 3, 4))

##ExterCond
train_x$ExterCond = factor(train_x$ExterCond, levels = c('Po', 'Fa', 'TA', 'Gd', 'Ex'),
                           labels = c(0, 1, 2, 3, 4))

##Foundation
train_x$Foundation = factor(train_x$Foundation, levels = c('Wood', 'Stone', 'Slab', 'PConc', 'CBlock', 'BrkTil'),
                            labels = c(0, 1, 2, 3, 4, 5))

##BsmtQual
train_x$BsmtQual = factor(train_x$BsmtQual, levels = c('No Basement', 'Po', 'Fa', 'TA', 'Gd', 'Ex'),
                          labels = c(0, 1, 2, 3, 4, 5))

##BsmtCond
train_x$BsmtCond = factor(train_x$BsmtCond, levels = c('No Basement', 'Po','Fa', 'TA', 'Gd', 'Ex'),
                          labels = c(0, 1, 2, 3, 4, 5))

##BsmtExposure
train_x$BsmtExposure = factor(train_x$BsmtExposure, levels = c('No Basement', 'No', 'Mn', 'Av', 'Gd'),
                              labels = c(0, 1, 2, 3, 4))

##BsmtFinType1
train_x$BsmtFinType1 = factor(train_x$BsmtFinType1, levels = c('No Basement', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'),
                              labels = c(0, 1, 2, 3, 4, 5, 6))

##BsmtFinType2
train_x$BsmtFinType2 = factor(train_x$BsmtFinType2, levels = c('No Basement', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'),
                              labels = c(0, 1, 2, 3, 4, 5, 6))

##Heating
train_x$Heating = factor(train_x$Heating, levels = c('Wall', 'OthW', 'Grav', 'GasW', 'GasA', 'Floor'),
                         labels = c(0, 1, 2, 3, 4, 5))

##HeatingQC
train_x$HeatingQC = factor(train_x$HeatingQC, levels = c('Po', 'Fa', 'TA', 'Gd', 'Ex'),
                           labels = c(0, 1, 2, 3, 4))

##CentralAir
train_x$CentralAir = factor(train_x$CentralAir, levels = c('N', 'Y'),
                            labels = c(0, 1))

##Electrical
train_x$Electrical = factor(train_x$Electrical, levels = c('Mix', 'FuseP', 'FuseF', 'FuseA', 'SBrkr'),
                            labels = c(0, 1, 2, 3, 4))

##KitchenQual
train_x$KitchenQual = factor(train_x$KitchenQual, levels = c('Po', 'Fa', 'TA', 'Gd', 'Ex'),
                             labels = c(0, 1, 2, 3, 4))

##Functional
train_x$Functional = factor(train_x$Functional, levels = c('Sal', 'Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ'),
                            labels = c(0, 1, 2, 3, 4, 5, 6, 7))

##FireplaceQu
train_x$FireplaceQu = factor(train_x$FireplaceQu, levels = c('No Fireplace', 'Po', 'Fa', 'TA', 'Gd', 'Ex'),
                             labels = c(0, 1, 2, 3, 4, 5))

##GarageType
train_x$GarageType = factor(train_x$GarageType, levels = c('No Garage', 'Detchd', 'CarPort', 'BuiltIn', 'Basment', 'Attchd', '2Types'),
                            labels = c(0, 1, 2, 3, 4, 5, 6))

##GarageFinish
train_x$GarageFinish = factor(train_x$GarageFinish, levels = c('No Garage', 'Unf', 'RFn', 'Fin'),
                             labels = c(0, 1, 2, 3))

##GarageQual
train_x$GarageQual = factor(train_x$GarageQual, levels = c('No Garage', 'Po', 'Fa', 'TA', 'Gd', 'Ex'),
                            labels = c(0, 1, 2, 3, 4, 5))

##GarageCond
train_x$GarageCond = factor(train_x$GarageCond, levels = c('No Garage', 'Po', 'Fa', 'TA', 'Gd', 'Ex'),
                            labels = c(0, 1, 2, 3, 4, 5))

##PavedDrive
train_x$PavedDrive = factor(train_x$PavedDrive, levels = c('N', 'P', 'Y'),
                            labels = c(0, 1, 2))

##PoolQC
train_x$PoolQC = factor(train_x$PoolQC, levels = c('No Pool', 'Fa', 'TA', 'Gd', 'Ex'),
                        labels = c(0, 1, 2, 3, 4))

##Fence
train_x$Fence = factor(train_x$Fence, levels = c('No Fence', 'MnWw', 'GdWo', 'MnPrv', 'GdPrv'),
                       labels = c(0, 1, 2, 3, 4))

##MiscFeature
train_x$MiscFeature = factor(train_x$MiscFeature, levels = c('None', 'TenC', 'Shed', 'Othr', 'Gar2', 'Elev'),
                             labels = c(0, 1, 2, 3, 4, 5))

##SaleType
train_x$SaleType = factor(train_x$SaleType, levels = c('Oth', 'ConLD', 'ConLI', 'ConLw', 'Con', 'COD', 'New', 'VWD', 'CWD', 'WD'),
                          labels = c(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))

##SaleCondition
train_x$SaleCondition = factor(train_x$SaleCondition, levels = c('Partial', 'Family', 'Alloca', 'AdjLand', 'Abnorml', 'Normal'),
                               labels = c(0, 1, 2, 3, 4, 5))

##check again
str(train_x)

colnames(train_x)[colSums(is.na(train_x)) > 0]

##MAKE CERTAIN NUMERIC VARIABLES FACTOR
##OverallQual
train_x$OverallQual = as.factor(train_x$OverallQual)

##OverallCond
train_x$OverallCond = as.factor(train_x$OverallCond)

##BsmtFullBath
train_x$BsmtFullBath = as.factor(train_x$BsmtFullBath)

##BsmtHalfBath
train_x$BsmtHalfBath = as.factor(train_x$BsmtHalfBath)

##FullBath
train_x$FullBath = as.factor(train_x$FullBath)

##HalfBath
train_x$HalfBath = as.factor(train_x$HalfBath)

##BedroomAbvGr
train_x$BedroomAbvGr = as.factor(train_x$BedroomAbvGr)

##KitchenAbvGr
train_x$KitchenAbvGr = as.factor(train_x$KitchenAbvGr)

##TotRmsAbvGrd
train_x$TotRmsAbvGrd = as.factor(train_x$TotRmsAbvGrd)

##Fireplaces
train_x$Fireplaces = as.factor(train_x$Fireplaces)

##GarageCars
train_x$GarageCars = as.factor(train_x$GarageCars)

str(train_x)

##Scale the variables
scaling = train_x %>% select(LotFrontage, LotArea, MasVnrArea, BsmtFinSF1, BsmtFinSF2,BsmtUnfSF, TotalBsmtSF, X1stFlrSF, X2ndFlrSF, 
                             LowQualFinSF, GrLivArea, GarageArea, WoodDeckSF, OpenPorchSF, EnclosedPorch, X3SsnPorch, ScreenPorch,
                             PoolArea, MiscVal)

PreNum = preProcess(scaling, method = c('center', 'scale'))
print(PreNum)

train_x = predict(PreNum, train_x)
dim(train_x)

numVars = which(sapply(train_x, is.numeric))
factorVars = which(sapply(train_x, is.factor))

cat('There are', length(numVars), 'numeric variables, and', length(factorVars), 'factor variables')

##First Correlation Visualization
matrix = model.matrix(~0+., data = train_x) %>%
  cor(use = 'pairwise.complete.obs')
matrix_sorted = as.matrix(sort(matrix[, 'SalePrice'], decreasing = T))
MatrixHigh = names(which(apply(matrix_sorted, 1, function(x) abs(x)>0.4)))
cor_factor = matrix[MatrixHigh, MatrixHigh]
corrplot.mixed(cor_factor, tl.col = 'black', tl.pos = 'lt', tl.cex = 0.7, cl.cex = .7, number.cex = .7)

#Correlation without factor variables
numericVars = which(sapply(train_x, is.numeric))
numericVarNames = names(numericVars) 
train_x_numVars = train_x[, numericVars]
cor_numVars = cor(train_x_numVars, use = 'pairwise.complete.obs') #correlations of all numeric variables
#sort on decreasing correlations with SalePrice
cor_sorted = as.matrix(sort(cor_numVars[, 'SalePrice'], decreasing = T))
#select only high correlations
CorHigh = names(which(apply(cor_sorted, 1, function(x) abs(x) > 0.5)))
cor_numVars = cor_numVars[CorHigh, CorHigh]
corrplot.mixed(cor_numVars, tl.col = 'black', tl.pos = 'lt')
#####-----##########-----##########-----##########-----##########-----##########-----##########-----##########-----##########-----#####
##Drop variables with few distinction
##DROP Street
train_x$Street = NULL

##DROP Condition2
train_x$Condition2 = NULL

##DROP RoofMatl
train_x$RoofMatl = NULL

##DROP Heating
train_x$Heating = NULL

##DROP LowQualFinSF
train_x$LowQualFinSF = NULL

##DROP X3SsnPorch
train_x$X3SsnPorch = NULL

#DROP PoolQC
train_x$PoolQC = NULL

#DROP PoolArea
train_x$PoolArea = NULL

##Second Correlation Visualization
matrix = model.matrix(~0+., data = train_x) %>%
  cor(use = 'pairwise.complete.obs')
matrix_sorted = as.matrix(sort(matrix[, 'SalePrice'], decreasing = T))
MatrixHigh = names(which(apply(matrix_sorted, 1, function(x) abs(x)>0.5)))
cor_factor = matrix[MatrixHigh, MatrixHigh]
corrplot.mixed(cor_factor, tl.col = 'black', tl.pos = 'lt', tl.cex = 0.7, cl.cex = .7, number.cex = .7)

##Plots
#IsNew
ggplot(train_x[!is.na(train_x$SalePrice), ], aes(x=IsNew, y=SalePrice)) +
  geom_bar(stat = 'summary', fun ='mean', fill='blue') +
  geom_label(stat = "count", aes(label = ..count.., y = ..count..), size=6) +
  scale_y_continuous(breaks= seq(0, 800000, by=50000)) +
  theme_grey(base_size = 18) +
  geom_hline(yintercept=163000, linetype="dashed")

#BldgType
ggplot(train_x[!is.na(train_x$SalePrice), ], aes(x=BldgType, y=SalePrice)) +
  geom_bar(stat = 'summary', fun ='mean', fill='blue') +
  geom_label(stat = "count", aes(label = ..count.., y = ..count..), size=6) +
  scale_y_continuous(breaks= seq(0, 800000, by=50000)) +
  theme_grey(base_size = 18) +
  geom_hline(yintercept=163000, linetype="dashed")

#YearBuilt
ggplot(train_x[!is.na(train_x$SalePrice), ], aes(x=YearBuilt, y=SalePrice)) +
  geom_point(color = 'steelblue') + geom_smooth(method = 'lm', color = 'black') +
  scale_y_continuous(breaks= seq(0, 800000, by=50000))

#TotRmsAbvGrd
ggplot(train_x[!is.na(train_x$SalePrice), ], aes(TotRmsAbvGrd, SalePrice)) + 
  geom_bar(stat = 'summary', fun = 'mean', fill = 'steelblue') +
  geom_label(stat = 'count', aes(label = ..count.., y = ..count..)) +
  scale_y_continuous(breaks = seq(0, 350000, by = 50000)) +
  theme_grey(base_size = 16) +
  geom_hline(yintercept = median(train_x$SalePrice), linetype = 'dashed')

#BedroomAbvGr
ggplot(train_x[!is.na(train_x$SalePrice), ], aes(BedroomAbvGr, SalePrice)) + 
  geom_bar(stat = 'summary', fun = 'mean', fill = 'steelblue') +
  geom_label(stat = 'count', aes(label = ..count.., y = ..count..)) +
  scale_y_continuous(breaks = seq(0, 350000, by = 50000)) +
  theme_grey(base_size = 16) +
  geom_hline(yintercept = median(train_x$SalePrice), linetype = 'dashed')

#GrLivArea
ggplot(train_x[!is.na(train_x$SalePrice), ], aes(x=GrLivArea, y=SalePrice))+
  geom_point(col='blue') + geom_smooth(method = "lm", se=FALSE, color="black") +
  scale_y_continuous(breaks= seq(0, 800000, by=100000)) +
  geom_text_repel(aes(label = ifelse(GrLivArea > 6, rownames(train_x), '')))

#check
numVars = which(sapply(train_x, is.numeric))
factorVars = which(sapply(train_x, is.factor))

cat('There are', length(numVars), 'numeric variables, and', length(factorVars), 'factor variables')

###########################################################
#One Hot Encode
datafor_hot_names = names(factorVars)
data_hot = train_x[, datafor_hot_names]

dummies = dummyVars(" ~. ", data = data_hot)
dummies = data.frame(predict(dummies, newdata = data_hot))

#check how many variables have less than 20 observations and remove (i choose 20 because one variable can have, for example, 13 in train
#and 5 in test. i want that every variable have at least 10 observations, so i think 20 is a good number for that conditions.)

few = which(colSums(dummies[1:nrow(dummies), ], ) <= 20)
fewnames = names(few)

#remove
dummies = dummies[, -few]

numericVarNames = names(numVars)
data_num = train_x[, numericVarNames]

final_data = cbind(dummies, data_num)

#Make Train and Test Set again
x.train = data.matrix(final_data[1:1460, 1:252])
x.test = data.matrix(final_data[1461:2919, 1:252])

y.train = final_data$SalePrice[1:1460]

qqPlot(y.train)
#Make Ridge Regression
ridge.fit = cv.glmnet(x.train, y.train, type.measure = 'mse', alpha = 0, family = 'gaussian')
coef(ridge.fit, s= 'lambda.min') 
ridge.pred = predict(ridge.fit, s = ridge.fit$lambda.1se, newx = x.test)

ridge.fit$lambda.min
plot(ridge.fit)
#Make Lasso Regression
lasso.fit = cv.glmnet(x.train, y.train, type.measure = 'mse', alpha = 1, family = 'gaussian')
coef(lasso.fit, s= 'lambda.min')
lasso.pred = predict(lasso.fit, s = lasso.fit$lambda.1se, newx = x.test)

###
lassoVarImp = varImp(lasso.fit1, scale = F, lambda = 710.1178)
lasso.fit1 = glmnet(x.train, y.train, type.measure = 'mse', alpha = 1, family = 'gaussian')
coef(lasso.fit1, s = lasso.fit$lambda.min)
###

#Make Elastic Net Regression
#Try only one alpha - 0.5
elastic.fit = cv.glmnet(x.train, y.train, type.measure = 'mse', alpha = 0.5, family = 'gaussian')
coef(elastic.fit)
elastic.pred = predict(elastic.fit, s = elastic.fit$lambda.1se, newx = x.test)

#Note: Since i don't have x.test i can't make the goodness of fit. So i need to split the train dataset, make the regressions again and 
#compare the predict with real values.

datasplit = final_data[1:1460, ]
set.seed(3)
sample = sample.split(datasplit$SalePrice, SplitRatio = 0.7)
train_for = subset(datasplit, sample == T)
test_for = subset(datasplit, sample == F)

x_train = data.matrix(train_for[, 1:252])
x_test = data.matrix(test_for[, 1:252])

y_train = train_for$SalePrice
y_test = test_for$SalePrice

#Ridge
alpha0.fit = cv.glmnet(x_train, y_train, type.measure = 'mse', alpha = 0, family = 'gaussian')
coef(alpha0.fit)
pred.alpha0 = predict(alpha0.fit, s = alpha0.fit$lambda.min, newx = x_test)

#SST and SSE
sst = sum((y_test - mean(y_test))^2)
sse.alpha0 = sum((pred.alpha0 - y_test)^2)
mean((y_test - pred.alpha0)^2)

#find R-Squared
rsq.alpha0 <- 1 - sse.alpha0/sst
rsq.alpha0

#Lasso
alpha1.fit = cv.glmnet(x_train, y_train, type.measure = 'mse', alpha = 1, family = 'gaussian')
coef(alpha1.fit)
pred.alpha1 = predict(alpha1.fit, s = alpha1.fit$lambda.min, newx = x_test)

#SST and SSE
sse.alpha1 = sum((pred.alpha1 - y_test)^2)
mean((y_test - pred.alpha1)^2)

#find R-Squared
rsq.alpha1 <- 1 - sse.alpha1/sst
rsq.alpha1

#Elastic Net
#Try only one alpha - 0.5
alpha0.5.fit = cv.glmnet(x_train, y_train, type.measure = 'mse', alpha = 0.5, family = 'gaussian')
coef(alpha0.5.fit)
pred.alpha0.5 = predict(alpha0.5.fit, s = alpha0.5.fit$lambda.min, newx = x_test)

#SST and SSE
sse.alpha0.5 = sum((pred.alpha0.5 - y_test)^2)
mean((y_test - pred.alpha0.5)^2)

#find R-Squared
rsq.alpha0.5 <- 1 - sse.alpha0.5/sst
rsq.alpha0.5

#Now try find the best alpha
list.of.fits = list()
for (i in 0:10) {
  model.name = paste0("alpha", i/10)
  
  set.seed(123)
  list.of.fits[[model.name]] = 
    cv.glmnet(x_train, y_train, type.measure = 'mse', alpha = i/10, family = 'gaussian')
}

results = data.frame()
for (i in 0:10) {
  model.name = paste0("alpha", i/10)
  
  predicted = predict(list.of.fits[[model.name]], s = list.of.fits[[model.name]]$lambda.min, newx = x_test)
  mse = mean((y_test - predicted)^2)
  temp = data.frame(alpha = i/10, mse = mse, model.name = model.name)
  results = rbind(results, temp)
}

results

#Compare with other regressions
#Linear Regression
linear.fit = lm(SalePrice ~. -1, data = train_for)
summary(linear.fit)
pred.linear = predict(linear.fit, newdata = test_for)

sse.linear = sum((pred.linear - test_for$SalePrice)^2)
sse.linear
rsq.linear = 1 - sse.linear/sst
rsq.linear

mean((test_for$SalePrice - pred.linear)^2)

a = sum(linear.fit$residuals^2) #sse
b = sum((train_for$SalePrice - mean(train_for$SalePrice))^2) #sst
1 -a/b
mean(linear.fit$residuals^2)

#SVM Regression
svm.fit = svm(train_for$SalePrice ~., data = train_for, type = 'eps-regression')

pred.svm = predict(svm.fit, newdata = test_for)

sse.svm = sum((pred.svm - test_for$SalePrice)^2)
sse.svm
rsq.svm = 1 - sse.svm/sst
rsq.svm

mean((test_for$SalePrice - pred.svm)^2)

sum((svm.fit$residuals)^2)

