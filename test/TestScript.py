import pandas as pd
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import os
import math

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

y = train.SalePrice
#np.log() # to take logs of target variable

dtrain = xgb.DMatrix(X_train, label=y)
dtest = xgb.DMatrix(X_test)

params = {"max_depth": 2, "eta": 0.1}
model = xgb.cv(params, dtrain, num_boost_round=500, early_stopping_rounds=100) #Question what is CV about this? Surely we shouldn't be able to plot a single batch run?

print(model.columns)
print("Min train rmse:" + str(min(model['train-rmse-mean'])))
print("Min test rmse:" + str(min(model['test-rmse-mean'])))
#math.sqrt math.exp

#Fig1. plotting from training sample 30 onwards only eliminates visualising the initial large error drop
p = model.loc[30:, ["test-rmse-mean", "train-rmse-mean"]].plot()

#Fig2
plt.figure(figsize=(15, 10))
plt.plot(model.index, model['train-rmse-mean'], 'r')
plt.plot(model.index, model['test-rmse-mean'], 'b')
#plt.ylim(0.8, 0.9)
#plt.xlim(1000,55000)
plt.xlabel('# training examples')
plt.ylabel('rmse')
plt.legend(['Training Set', 'CV set'], loc='lower right')
plt.show()

#plt.scatter(x=model['train-rmse-mean'], y=model['test-rmse-mean'])
#plt.show()




