import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import cross_validate
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import os
import matplotlib.pylab as plt
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

#https://stackoverflow.com/questions/27041724/using-conditional-to-generate-new-column-in-pandas-dataframe/27045135
#Create HasBsmt column

#all_data['HasBsmt'] = np.where(all_data['TotalBsmtSF'] > 0, 1, 0)
#all_data.loc[all_data['HasBsmt'] == 1, 'HasBsmt'] = np.log1p(all_data['TotalBsmtSF'])

train['HasBsmt'] = np.where(train['TotalBsmtSF'] == 0, 0, 1)
train.loc[train['HasBsmt'] == 1, 'TotalBsmtSF'] = np.log1p(train['TotalBsmtSF'])
train['TotalBsmtSF'] = np.where(train['TotalBsmtSF'] == 0, np.nan, train['TotalBsmtSF'])

all_data['GrLivArea'] = np.log1p(all_data['GrLivArea'])

##Missing Data: The below strategy of deleting columns increased the error by 5%
#total = all_data.isnull().sum().sort_values(ascending=False)
#percent = (all_data.isnull().sum()/all_data.isnull().count()).sort_values(ascending=False)
#missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
#missing_data.head(20)
##dealing with missing data. Drop Col if > 15% missing data.
#all_data = all_data.drop((missing_data[missing_data['Percent'] > 0.15]).index, 1)
#all_data = all_data.drop(all_data.loc[all_data['Electrical'].isnull()].index)
#all_data.isnull().sum().max() #just checking that there's no missing data missing...

print(all_data.dtypes)
all_data = pd.get_dummies(all_data)
print(all_data.shape)

print(all_data.head())
print(all_data.dtypes)

X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]

X_test.head()

y = np.log1p(train.SalePrice)

def modelfit(alg, train, y, test, useTrainCV=True, cv_folds=5, early_stopping_rounds=50, test_preds=False):
    # list of metrics and params: https://xgboost.readthedocs.io/en/latest/parameter.html
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(train, label=y)
        xgtest = xgb.DMatrix(test)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=xgb_param['n_estimators'], nfold=cv_folds,
                          metrics='rmse', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])

        print("\n")
        print(cvresult.iloc[(alg.n_estimators-1)])

        #Fit algorithm on the data
        alg.fit(train, y, eval_metric='rmse')

        #Predict training set:
        dtrain_predictions = alg.predict(train)

        #Print model report:
        print("\nModel Report")
        print("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error(np.exp(y), np.exp(dtrain_predictions))))

        #feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
        #print(pd.DataFrame(alg.feature_importances_))
        #feat_imp = pd.DataFrame(alg.feature_importances_)
        #feat_imp.plot(kind='bar', title='Feature Importances')
        #plt.ylabel('Feature Importance Score')

        if test_preds:
            # Predict test set:
            dtest_predictions = alg.predict(test)
            return dtest_predictions, alg





#Step 1- Find the number of estimators for a high learning rate
#predictors = [x for x in train.columns if x not in [target, Id]]
xgb1 = XGBRegressor(
        learning_rate=0.1,
        n_estimators=1000,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        nthread=12,
        scale_pos_weight=1,
        seed=27)
modelfit(xgb1, X_train, y, X_test)

#Grid seach on subsample and max_features
#Choose all predictors except target & IDcols
param_test1 = {
    'max_depth': range(3, 10, 2),
    'min_child_weight': range(1, 6, 2)
}
gsearch1 = GridSearchCV(estimator=XGBRegressor(learning_rate=0.1, n_estimators=214, max_depth=5,
                                        min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                        nthread=12, scale_pos_weight=1, seed=27),
                       param_grid = param_test1, scoring="neg_mean_squared_error", n_jobs=4, iid=False, cv=5, return_train_score=True)
gsearch1.fit(X_train, y)
#Scoring:
#https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

#https://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html
#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV

print(gsearch1.best_estimator_)
print(gsearch1.best_params_)
print(gsearch1.best_score_)
print(gsearch1.best_index_)

print(gsearch1.cv_results_)

#Grid seach on subsample and max_features
param_test2 = {
    'max_depth':[4,5,6],
    'min_child_weight':[4,5,6]
}
gsearch2 = GridSearchCV(estimator=XGBRegressor(learning_rate=0.1, n_estimators=214, max_depth=5,
                                        min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                        nthread=12, scale_pos_weight=1, seed=27),
                       param_grid = param_test2, scoring="neg_mean_squared_error", n_jobs=4, iid=False, cv=5, return_train_score=True)
gsearch2.fit(X_train, y)
#https://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html

print(gsearch2.best_estimator_)
print(gsearch2.best_params_)
print(gsearch2.best_score_)
print(gsearch2.best_index_)

param_test2b = {
    'min_child_weight':[6, 8, 10, 12]
}
gsearch2b = GridSearchCV(estimator=XGBRegressor(learning_rate=0.1, n_estimators=214, max_depth=4,
                                        min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                        nthread=12, scale_pos_weight=1, seed=27),
                       param_grid = param_test2b, scoring="neg_mean_squared_error", n_jobs=4, iid=False, cv=5, return_train_score=True)
gsearch2b.fit(X_train, y)
#https://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html

print(gsearch2b.best_estimator_)
print(gsearch2b.best_params_)
print(gsearch2b.best_score_)
print(gsearch2b.best_index_)

param_test3 = {
    'gamma':[i/10.0 for i in range(0, 5)]
}
gsearch3 = GridSearchCV(estimator=XGBRegressor(learning_rate=0.1, n_estimators=214, max_depth=4,
                                        min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                        nthread=12, scale_pos_weight=1, seed=27),
                       param_grid = param_test3, scoring="neg_mean_squared_error", n_jobs=4, iid=False, cv=5, return_train_score=True)
gsearch3.fit(X_train, y)
#https://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html

print(gsearch3.best_estimator_)
print(gsearch3.best_params_)
print(gsearch3.best_score_)
print(gsearch3.best_index_)

xgb2 = XGBRegressor(
        learning_rate=0.1,
        n_estimators=1000,
        max_depth=4,
        min_child_weight=6,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        nthread=12,
        scale_pos_weight=1,
        seed=27)
modelfit(xgb2, X_train, y, X_test)

param_test4 = {
    'subsample': [i/10.0 for i in range(6, 10)],
    'colsample_bytree': [i/10.0 for i in range(6, 10)]
}

gsearch4 = GridSearchCV(estimator=XGBRegressor(learning_rate=0.1, n_estimators=214, max_depth=4,
                                        min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                        nthread=12, scale_pos_weight=1, seed=27),
                       param_grid = param_test4, scoring="neg_mean_squared_error", n_jobs=4, iid=False, cv=5, return_train_score=True)
gsearch4.fit(X_train, y)
#https://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html

print(gsearch4.best_estimator_)
print(gsearch4.best_params_)
print(gsearch4.best_score_)
print(gsearch4.best_index_)

param_test5 = {
    'subsample': [i/100.0 for i in range(65, 80, 5)],
    'colsample_bytree': [i/100.0 for i in range(85, 100, 5)]
}

gsearch5 = GridSearchCV(estimator=XGBRegressor(learning_rate=0.1, n_estimators=214, max_depth=4,
                                        min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                        nthread=12, scale_pos_weight=1, seed=27),
                       param_grid = param_test5, scoring="neg_mean_squared_error", n_jobs=4, iid=False, cv=5, return_train_score=True)
gsearch5.fit(X_train, y)
#https://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html

print(gsearch5.best_estimator_)
print(gsearch5.best_params_)
print(gsearch5.best_score_)
print(gsearch5.best_index_)

param_test7 = {
    'reg_alpha': [0, 0.001, 0.005, 0.01, 0.05]
}

gsearch7 = GridSearchCV(estimator=XGBRegressor(learning_rate=0.1, n_estimators=214, max_depth=4,
                                        min_child_weight=6, gamma=0, subsample=0.75, colsample_bytree=0.9,
                                        nthread=12, scale_pos_weight=1, seed=27),
                       param_grid = param_test7, scoring="neg_mean_squared_error", n_jobs=4, iid=False, cv=5, return_train_score=True)
gsearch7.fit(X_train, y)
#https://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html

print(gsearch7.best_estimator_)
print(gsearch7.best_params_)
print(gsearch7.best_score_)
print(gsearch7.best_index_)


xgb3 = XGBRegressor(
        learning_rate=0.1,
        n_estimators=1000,
        max_depth=4,
        min_child_weight=6,
        gamma=0,
        subsample=0.75,
        colsample_bytree=0.9,
        reg_alpha=0.001,
        nthread=12,
        scale_pos_weight=1,
        seed=27)
modelfit(xgb3, X_train, y, X_test)

xgb4 = XGBRegressor(
        learning_rate=0.01,
        n_estimators=5000,
        max_depth=4,
        min_child_weight=6,
        gamma=0,
        subsample=0.75,
        colsample_bytree=0.9,
        reg_alpha=0.001,
        nthread=12,
        scale_pos_weight=1,
        seed=27)

predictions2, model = modelfit(xgb4, X_train, y, X_test, test_preds=True)
predictions2 = np.exp(predictions2)
submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predictions2})
submission.to_csv('submission3.csv', index=False)

feature_imp = model.get_booster().get_score(importance_type='gain')
print(feature_imp)
#lists = sorted(feature_imp.values, reverse=True)
#x, y = zip(*lists)
#plt.plot(x, y)
#plt.show()

xgb.plot_importance(model).show() # not working
#XGBRegressor.get_booster().get_score(importance_type='gain') #defualt is importance_type='weight'
#https://datascience.stackexchange.com/questions/34209/xgboost-quantifying-feature-importances