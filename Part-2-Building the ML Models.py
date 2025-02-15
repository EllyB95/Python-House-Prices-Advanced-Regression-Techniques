# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 19:25:15 2025

@author: harpr
"""

#%%
# BUILD MODELS
#1 LINEAR REGRESSION
lr = LinearRegression()
#%% 
#FIT THE DATA THROUGH THE LINEAR REGRESSION
lr.fit(X_train, y_train)

#%% 
# PREDICT THE OUTPUT
y_pred_lr = lr.predict(X_test)

#%%
# GETTING THE ERROR RESULTS 
mean_squared_error(y_test, y_pred_lr)

#%%
#2 Random Forest Regressor
RFR = RandomForestRegressor(random_state=13)

#%%
# BUILDING THE PARAMETER GRID FOR THE FOREST REGRESSION FOR BETTER RESULTS
param_grid_RFR = {
    'max_depth': [5, 10, 15],
    'n_estimators': [100, 250, 500],
    'min_samples_split': [3, 5, 10]
}

#%%
from sklearn.model_selection import GridSearchCV
#%%
rfr_cv = GridSearchCV(RFR, param_grid_RFR, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

#%%
#FITTING THE DATA TO rfr
rfr_cv.fit(X_train, y_train)
#%%
# GET THE SCORE
np.sqrt(-1 * rfr_cv.best_score_)
#%%
rfr_cv.best_params_
#%%
from xgboost import XGBRegressor
#%%
XGB = XGBRegressor(random_state=13)
#%%
param_grid_XGB = {
    'learning_rate': [0.05, 0.1, 0.2],
    'n_estimators': [300],
    'max_depth': [3],
    'min_child_weight': [1,2,3],
    'gamma': [0, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
}
#%%
xgb_cv = GridSearchCV(XGB, param_grid_XGB, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)

#%%
xgb_cv.fit(X_train, y_train)

#%%
np.sqrt(-1 * xgb_cv.best_score_)

#%%
ridge = Ridge()
#%%
param_grid_ridge = {
    'alpha': [0.05, 0.1, 1, 3, 5, 10],
    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag']
}
#%%
ridge_cv = GridSearchCV(ridge, param_grid_ridge, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
#%%
ridge_cv.fit(X_train, y_train)
#%%
np.sqrt(-1 * ridge_cv.best_score_)
#%%
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

#%%
GBR = GradientBoostingRegressor()
#%%
param_grid_GBR = {
    'max_depth': [12, 15, 20],
    'n_estimators': [200, 300, 1000],
    'min_samples_leaf': [10, 25, 50],
    'learning_rate': [0.001, 0.01, 0.1],
    'max_features': [0.01, 0.1, 0.7]
}
#%%
GBR_cv = GridSearchCV(GBR, param_grid_GBR, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
#%%
GBR_cv.fit(X_train, y_train)
#%%
np.sqrt(-1 * GBR_cv.best_score_)
#%%
import lightgbm as lgb
#%%
lgbm_regressor = lgb.LGBMRegressor()
#%%
param_grid_lgbm = {
    'boosting_type': ['gbdt', 'dart'],
    'num_leaves': [20, 30, 40],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300]
}

#%%
lgbm_cv = GridSearchCV(lgbm_regressor, param_grid_lgbm, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
#%%
lgbm_cv.fit(X_train, y_train)
#%%
np.sqrt(-1 * lgbm_cv.best_score_)
#%%
catboost = CatBoostRegressor(loss_function='RMSE', verbose=False)

#%%
param_grid_cat ={
    'iterations': [100, 500, 1000],
    'depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.5]
}
#%%
cat_cv = GridSearchCV(catboost, param_grid_cat, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
#%%
#%%
cat_cv.fit(X_train, y_train)
#%%
np.sqrt(-1 * cat_cv.best_score_)
#%%
vr = VotingRegressor([('gbr', GBR_cv.best_estimator_),
                      ('xgb', xgb_cv.best_estimator_),
                      ('ridge', ridge_cv.best_estimator_)],
                    weights=[2,3,1])
#%%
vr.fit(X_train, y_train)

#%%
y_pred_vr = vr.predict(X_test)

#%%
mean_squared_error(y_test, y_pred_vr, squared=False)


#%%
estimators = [
    ('gbr', GBR_cv.best_estimator_),
    ('xgb', xgb_cv.best_estimator_),
    ('cat', cat_cv.best_estimator_),
    ('lgb', lgbm_cv.best_estimator_),
    ('rfr', rfr_cv.best_estimator_),
]
#%%
stackreg = StackingRegressor(
            estimators = estimators,
            final_estimator = vr
)
#%%
stackreg.fit(X_train, y_train)
#%%
y_pred_stack = stackreg.predict(X_test)
#%%
mean_squared_error(y_test, y_pred_stack, squared=False)

#%%
df_test_preprocess = pipeline.transform(test_df)

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%