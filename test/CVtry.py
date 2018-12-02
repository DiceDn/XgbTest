import pandas as pd
import xgboost as xgb
import numpy as np  # use numpy for math functions.
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

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

y = np.log1p(train.SalePrice)

# For real-valued input, log1p is accurate also for x so small that 1 + x == 1 in floating-point accuracy.
def rmsle(ypred, ytest):
    assert len(ytest) == len(ypred)
    return np.sqrt(np.mean((np.log1p(ypred) - np.log1p(ytest)) ** 2))


dtrain = xgb.DMatrix(X_train, label=y)
dtest = xgb.DMatrix(X_test.iloc[0:1, ])

y2 = [200000]

print(dtest)


# Examine if file was correctly processed:
print("Train dataset contains {0} rows and {1} columns".format(dtrain.num_row(), dtrain.num_col()))
print("Test dataset contains {0} rows and {1} columns".format(dtest.num_row(), dtest.num_col()))

# use cross validation rather than a completely separate validation set.
params = {"max_depth": 2, "eta": 0.1, "silent": 1}
model = xgb.cv(params, dtrain, num_boost_round=5000, early_stopping_rounds=100)
               #verbose_eval=50)
#xgb.cv
#need to add watchlist to train, not auto cv, it will be ignored!
#The respective CV folds ARE the watchlist!
#xgb.cv is used to tune HPs and only returns the cv history of the process.
#use xgb.train to return a model.


xTrain, xTest, yTrain, yTest = train_test_split(X_train, y, test_size=0.2, random_state=0)
d_Train = xgb.DMatrix(xTrain, label=yTrain)
d_Test = xgb.DMatrix(yTest, label=xTest)

# rmse is the root mean squared error hence ^.5 has been done bringing units into the range of dependant variable.
print(model.columns)
print("Min train rmse:" + str(min(model['train-rmse-mean'])))
print("Min test rmse:" + str(min(model['test-rmse-mean'])))
print("Min Stopping Index:" + str(max(model.index)))
# print(model.dtype)
print("test rmse at round 50 : " + str(model.iloc[50]))
print("test rmse at round 100: " + str(model.iloc[100]))
print("test rmse at round 200 : " + str(model.iloc[200]))
print("test rmse at round 300 : " + str(model.iloc[300]))
print("test rmse at round 500 : " + str(model.iloc[500]))
print("test rmse at round " + str(max(model.index)) + " " + str(model.iloc[max(model.index)]))
# math.sqrt math.exp

# Fig1. plotting from training sample 30 onwards only eliminates visualising the initial large error drop
p = model.loc[30:, ["test-rmse-mean", "train-rmse-mean"]].plot()

# Fig2
plt.figure(figsize=(15, 10))
plt.plot(model.index, model['train-rmse-mean'], 'r')
plt.plot(model.index, model['test-rmse-mean'], 'b')
# plt.ylim(0.8, 0.9)
# plt.xlim(1000,55000)
plt.xlabel('# training examples')
plt.ylabel('rmse')
plt.legend(['Training Set', 'CV set'], loc='lower right')
plt.show()

# plt.scatter(x=model['train-rmse-mean'], y=model['test-rmse-mean'])
# plt.show()

bst()
print()
#res = mod.predict(dtest)# np.exp()
#print(res.dtype)

# df = pd.concat(a,b)
#df = pd.DataFrame(res)
#df.to_csv('submission1.csv', index=True, header=['Id', 'SalePrice'])
