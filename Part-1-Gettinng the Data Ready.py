# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 19:23:38 2025

@author: harpr
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 15:04:58 2025

@author: harpr
"""
#%% 
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder

from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRFRegressor
from catboost import CatBoostRegressor
from sklearn.kernel_ridge import KernelRidge

#%%
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('train.csv')

#%%
train_df.columns
train_df.describe()
#%% Check the DataTypes
Data_Types = train_df.dtypes[train_df.dtypes !='object']

#%% WE DROP THE OUTLIER TO INCREASE THE DATA QUALITY 
#Check for any outliers for MSSub Class
plt.scatter(x='MSSubClass', y='SalePrice', data = train_df)
plt.show()

#%% Check for any outliers for lot frontage. ie there are no lots bigger than 200 sqft, more than that will be outliers
plt.scatter(x='LotFrontage', y='SalePrice', data = train_df)
plt.show()

#%% Query is very good tool in Python to look for specific Data
train_df.query('LotFrontage>300')
#Drop 935 and 1299

#%% Check for any outliers for LotArea
plt.scatter(x='LotArea', y='SalePrice', data = train_df)
plt.show()

#%%
train_df.query('LotArea>50000')
# Drop 250, 314, 336, 707, 1397

#%% We Can Do these computations by cheacking the zscore as well. more the z score bigger the Outlier
#We usually DROP the bigger Zscore Rows
stats.zscore(train_df['LotArea']).sort_values().tail(10)

#%% Check for any outliers for OverallQuality
plt.scatter(x='OverallQual', y='SalePrice', data = train_df)
plt.show()

#%% 
train_df.query('OverallQual==10')

#%% Check for any outliers for Overall Condition
plt.scatter(x='OverallCond', y='SalePrice', data = train_df)
plt.show()

#%%
train_df.query('OverallCond == 5 & SalePrice> 700000')
#Drop 1183 
#%%
train_df.query('OverallCond == 6 & SalePrice> 700000')
#Drop 692
#%% Check Outliers for Year Built
plt.scatter(x='YearBuilt', y= 'SalePrice', data = train_df)
plt.show()

#%% Check Outliers for Year Remodeling
train_df.query('YearBuilt < 1900 & SalePrice> 400000')
#Drop 186
#%% 
plt.scatter(x='YearRemodAdd', y= 'SalePrice', data = train_df)
plt.show()

#%%
train_df.query('YearRemodAdd < 1970 & SalePrice> 300000')
train_df.query('YearRemodAdd < 2000 & SalePrice> 700000')
#Drop 314
#%% Check Outliers for Mass VNR Area
plt.scatter(x='MasVnrArea', y= 'SalePrice', data = train_df)
plt.show()

#%%
train_df.query('MasVnrArea > 1450')
#Drop 298
#%% Check Outliers for Mass Basement 1 finishing Area
plt.scatter(x='BsmtFinSF1', y= 'SalePrice', data = train_df)
plt.show()

#%%
train_df.query('BsmtFinSF1 > 5000')
#Drop 1299
#%% Check Outliers for Mass Basement 2 finishing Area
plt.scatter(x='BsmtFinSF2', y= 'SalePrice', data = train_df)
plt.show()

#%%
train_df.query('BsmtFinSF2 >400 & SalePrice > 500000')
#Drop 441
#%% Check Outliers for Mass Basement SQFT
plt.scatter(x='BsmtUnfSF', y= 'SalePrice', data = train_df)
plt.show()

#%% Check Outliers for Mass Basement Total SQFT
plt.scatter(x='TotalBsmtSF', y= 'SalePrice', data = train_df)
plt.show()
#%%
train_df.query('TotalBsmtSF > 5000')
#%% Check Outliers for 1st Floor SQFT
plt.scatter(x='1stFlrSF', y= 'SalePrice', data = train_df)
plt.show()

#%% Check Outliers for 1st Floor SQFT
plt.scatter(x='2ndFlrSF', y= 'SalePrice', data = train_df)
plt.show()

#%% Check Outliers for Low Quality SQFT
plt.scatter(x='LowQualFinSF', y= 'SalePrice', data = train_df)
plt.show()

#%%
train_df.query('LowQualFinSF > 550')
#Drop 186
#%% Check Outliers for Garage Living Area
plt.scatter(x='GrLivArea', y= 'SalePrice', data = train_df)
plt.show()

#%%
train_df.query('GrLivArea > 4400')
#Drop 524, 1299
#%% Check Outliers for bsmtFull Bath
plt.scatter(x='BsmtFullBath', y= 'SalePrice', data = train_df)
plt.show()

#%%
train_df.query('BsmtFullBath > 2.5')
#Drop 739
#%% Check Outliers for bsmtHalf Bath
plt.scatter(x='BsmtHalfBath', y= 'SalePrice', data = train_df)
plt.show() 

#%%
train_df.query('BsmtHalfBath > 1.75')
#Drop 598,955
#%%Check Outliers for Full Bath
plt.scatter(x='FullBath', y= 'SalePrice', data = train_df)
plt.show()

#%% Check Outliers for Half Bath
plt.scatter(x='HalfBath', y= 'SalePrice', data = train_df)
plt.show()

#%% Check Outliers for Bedroom Above Garage
plt.scatter(x='BedroomAbvGr', y= 'SalePrice', data = train_df)
plt.show()

#%%
train_df.query('BedroomAbvGr ==8 ')
#Drop 636
#%% Check Outliers for Kitchen Above Garage
plt.scatter(x='KitchenAbvGr', y= 'SalePrice', data = train_df)
plt.show()

#%%
train_df.query('KitchenAbvGr ==3 ')
#Drop 49, 810
#%% Check Outliers for Total Room Above ground
plt.scatter(x='TotRmsAbvGrd', y= 'SalePrice', data = train_df)
plt.show()

#%%
train_df.query('TotRmsAbvGrd ==14 ')
#Drop 636
#%% Check Outliers for Fireplaces
plt.scatter(x='Fireplaces', y= 'SalePrice', data = train_df)
plt.show()

#%% Check Outliers for Garage Year Built
plt.scatter(x='GarageYrBlt', y= 'SalePrice', data = train_df)
plt.show()

#%% Check Outliers for Garage Cars
plt.scatter(x='GarageCars', y= 'SalePrice', data = train_df)
plt.show()

#%% Check Outliers for Garage Area
plt.scatter(x='GarageArea', y= 'SalePrice', data = train_df)
plt.show()

#%%
train_df.query('GarageArea >1200 & SalePrice < 100000 ')
# Drop 1062
#%%Check Outliers for Wood Deck SQFT
plt.scatter(x='WoodDeckSF', y= 'SalePrice', data = train_df)
plt.show()

#%% Check Outliers for Open porch SQFT
plt.scatter(x='OpenPorchSF', y= 'SalePrice', data = train_df)
plt.show()

#%%
train_df.query('OpenPorchSF >500 & SalePrice<100000 ')
#Drop 496
#%% Check Outliers for OpenEnclosed porch SQFT
plt.scatter(x='EnclosedPorch', y= 'SalePrice', data = train_df)
plt.show()

#%%
train_df.query('EnclosedPorch >500 ')
#Drop 198
#%% Check Outliers for Three Season Porch
plt.scatter(x='3SsnPorch', y= 'SalePrice', data = train_df)
plt.show()

#%%Check Outliers for Screen Porch
plt.scatter(x='ScreenPorch', y= 'SalePrice', data = train_df)
plt.show()

#%%Check Outliers for Pool Area
plt.scatter(x='PoolArea', y= 'SalePrice', data = train_df)
plt.show()

#%%Check Outliers for Misc Val
plt.scatter(x='MiscVal', y= 'SalePrice', data = train_df)
plt.show()

#%%
train_df.query('MiscVal >14000 ')
#Drop 347
#%%Check Outliers for Month Sold
plt.scatter(x='MoSold', y= 'SalePrice', data = train_df)
plt.show()

#%%Check Outliers for Year Sold
plt.scatter(x='YrSold', y= 'SalePrice', data = train_df)
plt.show()

#%% Total Values we Got for OutLiers
values = [598, 955, 935, 1299, 250, 314, 336, 707, 379, 1183, 692, 186,
          441, 186, 524, 739, 598, 955, 636, 1062, 1191, 496, 198, 1338]

#%%
train_df = train_df[train_df.Id.isin(values)==False]
#%%
train_df = train_df[train_df.Id.isin(values)==False]
#%% NOW CHECK FOR THE NULL VALUES.
null_values = train_df.isnull().sum().sort_values(ascending=False).head(21)
#%%
train_df['MiscFeature'].unique()
#%%
train_df['Alley'].unique()
#%% FIX NULL VALUES OF ALLY
train_df['Alley'].fillna('No', inplace= True)
test_df['Alley'].fillna('No', inplace= True)
#%%FOR FURTHER ANALYSIS WE CAN DO CAT PLOT
sns.catplot(x="Alley", y="SalePrice", kind="box", data=train_df)
plt.show()
#%%
train_df['Fence'].unique()

#%%FIX NULL VALUES OF Fence
train_df['Fence'].fillna('No', inplace= True)
test_df['Fence'].fillna('No', inplace= True)

#%% CAT Plot For Fence
sns.catplot(x="Fence", y="SalePrice", kind="box", data=train_df)
plt.show()

#%%
train_df.query('Fence =="No"').count()
#%%
train_df['MasVnrType'].unique()
#%%FIX NULL VALUES OF MasVNR Type
train_df['MasVnrType'].fillna('No', inplace= True)
test_df['MasVnrType'].fillna('No', inplace= True)

#%% CAT Plot For MAsVnrType
sns.catplot(x="MasVnrType", y="SalePrice", kind="box", data=train_df)
plt.show()

#%% FIX NULL VALUES OF MasVNR Area
train_df['MasVnrArea'].fillna(0, inplace= True)
test_df['MasVnrArea'].fillna(0, inplace= True)

#%% FIX NULL VALUES OF Fire Place Quantity
train_df['FireplaceQu'].unique()

#%%
train_df['FireplaceQu'].fillna('No', inplace= True)
test_df['FireplaceQu'].fillna('No', inplace= True)

#%%
sns.catplot(x="FireplaceQu", y="SalePrice", kind="box", data=train_df)
plt.show()


#%% FIX NULL VALUES OF Fire Place Quantity
train_df['LotFrontage'].unique()

#%%
train_df['LotFrontage'].fillna('No', inplace= True)
test_df['LotFrontage'].fillna('No', inplace= True)

#%% FIX NULL VALUES OF GARAGE YEAR BUILT
train_df['GarageYrBlt'].unique()

#%% WE HAVE TO CHECK THE CORELATION between the hous year built and garage year built
#MOST OF THE TIME GARAGE IS BUILT WHEN HOUSE IS BUILT
# IF WE HAVE HIGH CORELATION THAT MEANS BOTH ARE SIMILAR. AND WE CAN END UP DROPING IT
train_df['GarageYrBlt'].corr(train_df['YearBuilt'])

#%% FIX NULL VALUES OF GARAGE CONDITION
train_df['GarageCond'].unique()

#%%
train_df['GarageCond'].fillna('No', inplace= True)
test_df['GarageCond'].fillna('No', inplace= True)

#%%
sns.catplot(x="GarageCond", y="SalePrice", kind="box", data=train_df)
plt.show()

#%% FIX NULL VALUES OF GARAGE Type
train_df['GarageType'].unique()

#%%
train_df['GarageType'].fillna('No', inplace= True)
test_df['GarageType'].fillna('No', inplace= True)

#%%
sns.catplot(x="GarageType", y="SalePrice", kind="box", data=train_df)
plt.show()

#%% FIX NULL VALUES OF GARAGE Finish
train_df['GarageFinish'].unique()

#%%
train_df['GarageFinish'].fillna('No', inplace= True)
test_df['GarageFinish'].fillna('No', inplace= True)

#%%
sns.catplot(x="GarageFinish", y="SalePrice", kind="box", data=train_df)
plt.show()

#%% FIX NULL VALUES OF GARAGE Quality
train_df['GarageQual'].unique()
#%% 
train_df['GarageQual'].fillna('Po', inplace= True)
test_df['GarageQual'].fillna('Po', inplace= True)

#%%
sns.catplot(x="GarageQual", y="SalePrice", kind="box", data=train_df)
plt.show()

#%% FIX NULL VALUES OF Basement fin type 2
train_df['BsmtFinType2'].unique()

#%% 
train_df['BsmtFinType2'].fillna('Unf', inplace= True)
test_df['BsmtFinType2'].fillna('Unf', inplace= True)

#%%
sns.catplot(x="BsmtFinType2", y="SalePrice", kind="box", data=train_df)
plt.show()

#%% FIX NULL VALUES OF Basement Exposer
train_df['BsmtExposure'].unique()

#%% 
train_df['BsmtExposure'].fillna('No', inplace= True)
test_df['BsmtExposure'].fillna('No', inplace= True)

#%%
sns.catplot(x="BsmtExposure", y="SalePrice", kind="box", data=train_df)
plt.show()

#%% FIX NULL VALUES OF Basement Quality
train_df['BsmtQual'].unique()

#%% 
train_df['BsmtQual'].fillna('No', inplace= True)
test_df['BsmtQual'].fillna('No', inplace= True)

#%%
sns.catplot(x="BsmtQual", y="SalePrice", kind="box", data=train_df)
plt.show()

#%% FIX NULL VALUES OF Basement Condition
train_df['BsmtCond'].unique()

#%% 
train_df['BsmtCond'].fillna('No', inplace= True)
test_df['BsmtCond'].fillna('No', inplace= True)

#%%
sns.catplot(x="BsmtCond", y="SalePrice", kind="box", data=train_df)
plt.show()

#%% FIX NULL VALUES OF Basement Fin 1
train_df['BsmtFinType1'].unique()

#%% 
train_df['BsmtFinType1'].fillna('Unf', inplace= True)
test_df['BsmtFinType1'].fillna('Unf', inplace= True)

#%%
sns.catplot(x="BsmtFinType1", y="SalePrice", kind="box", data=train_df)
plt.show()

#%% FIX NULL VALUES OF MasVnrArea
train_df['MasVnrArea'].unique()

#%% 
train_df['MasVnrArea'].fillna(0, inplace= True)
test_df['MasVnrArea'].fillna(0, inplace= True)

#%% FIX NULL VALUES OF Electrical
train_df['Electrical'].unique()

#%% 
train_df['Electrical'].fillna('SBrkr', inplace= True)
test_df['Electrical'].fillna('SBrkr', inplace= True)

#%%
sns.catplot(x="Electrical", y="SalePrice", kind="box", data=train_df)
plt.show()
#%%
# NULL VALUES ARE FILLED 
null_values1 = train_df.isnull().sum().sort_values(ascending=False).head(21)

#%%
#Drop the Columns With High Null Values

#%% 
#DROP COLUMNS WHICH HAVE SIMILAR PROPERTIES, OR NOT NEED TO TRAIN MODEL
train_df = train_df.drop(columns=['PoolQC', 'MiscFeature', 'Alley', 'Fence', 
                                  'GarageYrBlt', 'GarageCond', 'BsmtFinType2'])
#%% 
test_df = test_df.drop(columns=['PoolQC','MiscFeature','Alley','Fence',
                                  'GarageYrBlt','GarageCond','BsmtFinType2'])
#%% FEATURE ENGINEERING

#%% Adding the house age 
train_df['houseage']= train_df['YrSold'] - train_df['YearBuilt']
test_df['houseage'] = test_df['YrSold'] - test_df['YearBuilt']

#%%  Adding the House Remodel Age
train_df['houseremodelage']= train_df['YrSold'] - train_df['YearRemodAdd']
test_df['houseremodelage'] = test_df['YrSold'] - test_df['YearRemodAdd']

#%%  Adding Total SQFT
train_df['totalsf']= train_df['1stFlrSF'] + train_df['2ndFlrSF'] + train_df['BsmtFinSF1'] + train_df['BsmtFinSF2'] 
test_df['totalsf'] = test_df['1stFlrSF'] + test_df['2ndFlrSF'] + test_df['BsmtFinSF1'] + test_df['BsmtFinSF2']

#%% Adding Total Area
train_df['totalarea'] = train_df['GrLivArea'] + train_df['TotalBsmtSF']
test_df['totalarea'] = test_df['GrLivArea'] + test_df['TotalBsmtSF']

#%% Adding Total Bathrooms
train_df['totalbaths'] = train_df['BsmtFullBath'] + train_df['FullBath'] + 0.5 * (train_df['BsmtHalfBath'] + train_df['HalfBath']) 
test_df['totalbaths'] = test_df['BsmtFullBath'] + test_df['FullBath'] + 0.5 * (test_df['BsmtHalfBath'] + test_df['HalfBath']) 

#%% Adding Total Porch SQFT
train_df['totalporchsf'] = train_df['OpenPorchSF'] + train_df['3SsnPorch'] + train_df['EnclosedPorch'] + train_df['ScreenPorch'] + train_df['WoodDeckSF']
test_df['totalporchsf'] = test_df['OpenPorchSF'] + test_df['3SsnPorch'] + test_df['EnclosedPorch'] + test_df['ScreenPorch'] + test_df['WoodDeckSF']

#%% DROP SOME COLUMNS
train_df = train_df.drop(columns=['Id','YrSold', 'YearBuilt', 'YearRemodAdd', 
                                  '1stFlrSF', '2ndFlrSF', 'BsmtFinSF1', 'BsmtFinSF2',
                                  'GrLivArea', 'TotalBsmtSF','BsmtFullBath',
                                  'FullBath', 'BsmtHalfBath', 'HalfBath',
                                  'OpenPorchSF', '3SsnPorch', 'EnclosedPorch', 
                                  'ScreenPorch','WoodDeckSF'])
test_df = test_df.drop(columns=['Id','YrSold', 'YearBuilt', 'YearRemodAdd', 
                                  '1stFlrSF', '2ndFlrSF', 'BsmtFinSF1', 'BsmtFinSF2',
                                  'GrLivArea', 'TotalBsmtSF','BsmtFullBath',
                                  'FullBath', 'BsmtHalfBath', 'HalfBath',
                                  'OpenPorchSF', '3SsnPorch', 'EnclosedPorch', 
                                  'ScreenPorch','WoodDeckSF'])
#%%GET THE CORELATION MATRIX HEATMAP
#WHEREEVER THE CORELATION IS HIGH(0.8 OR HIGHER) IN BETWEEN TWO COLUMNS, WE WANT TO DROP ONE OF THEM 
# TO GET THE GOOD TRAIN MODEL.
correlation_matrix =train_df.corr(numeric_only=True)
plt.figure(figsize=(20,12))
sns.heatmap(correlation_matrix, annot =True, cmap ='coolwarm', fmt =".2f" )
plt.show()
# Drop garage area or garage cars because they have high corelation
#%%
train_df = train_df.drop(columns=['GarageArea'])
test_df = test_df.drop(columns=['GarageArea'])

#%%
# WE ARE GOING TO TO USE ENCODERS FOR ENCODING THE DATA
#1 ODINAL ENCODING
#2 ONE HOT ENCODING
#%%
ode_cols = ['LotShape', 'LandContour','Utilities','LandSlope',  'BsmtQual', 
            'BsmtFinType1',  'CentralAir',  'Functional', 
           'FireplaceQu', 'GarageFinish', 'GarageQual', 'PavedDrive',
           'ExterCond', 'KitchenQual', 'BsmtExposure', 'HeatingQC','ExterQual', 
           'BsmtCond']


#%%
ohe_cols = ['Street', 'LotConfig','Neighborhood', 'Condition1', 'Condition2',
            'BldgType', 'HouseStyle', 'RoofStyle', 'Exterior1st', 'Exterior2nd',
           'MasVnrType','Foundation',  'Electrical',  'SaleType', 'MSZoning', 
           'SaleCondition', 'Heating', 'GarageType', 'RoofMatl']

#%%
num_cols = train_df.select_dtypes(include=['int64', 'float64']).columns
#%%
num_cols = num_cols.drop('SalePrice')

#%%
#BUILDING PIPELINES
num_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
#%%
ode_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('ode', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))#Transform of No to -1 as mentioned in code for unknown_value failed do manually have to remove column later
])

#%%
ohe_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

#%%
#COLUMN TRANSFORM
col_trans = ColumnTransformer(transformers=[
    ('num_p', num_pipeline, num_cols),
    ('ode_p', ode_pipeline, ode_cols),
    ('ohe_p', ohe_pipeline, ohe_cols),
    ],
    remainder='passthrough', 
    n_jobs=-1)

#%%
pipeline = Pipeline(steps=[
    ('preprocessing', col_trans)
    ])

#%%
#DROPING SALE PRICE
X = train_df.drop('SalePrice', axis=1)
y = train_df['SalePrice']
#%%
# PREPROCESSING MATRIX
X_preprocessed = pipeline.fit_transform(X)

#%%
#TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2,
                                                    random_state=25)
#%% REPLACING THE STRING "No" WITH THE -1
H = pd.DataFrame(X_train)
#X_train1 = H.drop(H.columns[193], axis =1)
H = H.applymap(lambda x: -1 if isinstance(x, str) else x)
X_train = H.to_numpy()

#%%
I = pd.DataFrame(X_test)
#X_test1 = I.drop(I.columns[193], axis =1)
I = I.applymap(lambda x: -1 if isinstance(x, str) else x)
X_test = I.to_numpy()
#%%

