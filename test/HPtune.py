import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import cross_validate
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import os
import matplotlib.pyloab as plt
#%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

os.getcwd()
os.chdir('C:/Users/Richard/OneDrive/My Documents/Machine Learning/kaggle/ames/')

train = pd.read_csv('train.csv', na_values='NA')
print(train.head())
test = pd.read_csv('test.csv', na_values='NA')
print(test.head())

all_data = pd.concat((train.loc[:, 'MSSubClass':'SaleCondition'], test.loc[:, 'MSSubClass':'SaleCondition']))

print(all_data.dtypes)
all_data = pd.get_dummies(all_data)
print(all_data.shape)

print(all_data.head())
print(all_data.dtypes)

X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]

X_test.head()

y = np.log1p(train.SalePrice)

def modelfit(alg, train, test, predictors, target, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    # list of metrics and params: https://xgboost.readthedocs.io/en/latest/parameter.html
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(train[predictors].values, label=train[target].values)
        xgtest = xgb.DMatrix(test[predictors].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=xgb_param['n_estimators'], nfold=cv_folds,
                          metrics='rmse', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])

        #Fit algorithm on the data
        alg.fit(train[predictors], train[target], eval_metric='rmse')

        #Predict training set:
        dtrain_predictions = alg.predict(train[predictors])

        #Print model report:
        print("\nModel Report")
        print("Accuracy : %.4g" % metrics.mean_squared_error(train[target].values,dtrain_predictions))

        feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')

        #Step 1- Find the number of estimators for a high learning rate
        predictors = [x for x in train.columns if x not in [target, Id]]
        xgb1 = XGBRegressor(
            learning_rate=0.1,
            n_estimators=1000,
            max_depth=5,
            min_child_weight=1,
            gamma=0,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='rmse',
            nthread=4,
            scale_pos_weight=1,
            seed=27)
        modelfit(xgb1, train, test, predictors)