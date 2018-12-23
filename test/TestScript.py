import numpy as np
import pandas as pd
import xgboost as xgb
import os
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


#watchList = [(d_Test, 'test'), (d_Train, 'train')]
#bst = xgb.train(params, d_Train, num_boost_round=5000, watchlist=watchList) #early stopping rounds clearly needs a validation set passed via the watchlist.

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


xTrain, xTest, yTrain, yTest = train_test_split(X_train, y, test_size=0.2, random_state=0)
#d_Train = xgb.DMatrix(xTrain, label=yTrain)
#d_Test = xgb.DMatrix(yTest, label=xTest)

mod1 = XGBRegressor(n_estimators=1000, learning_rate=0.05)
mod1.fit(xTrain, yTrain, early_stopping_rounds=100, eval_set=[(xTest, yTest)], verbose=True)
predictions = mod1.predict(xTest)

print("MSE = " + str(mean_squared_error(predictions, yTest)))

print(test.head())

print(predictions)
print(predictions.dtype)

predictions2 = np.exp(mod1.predict(X_test))
print(predictions2)
print(predictions2.dtype)

print(test.shape)
print(predictions2.shape)

#Set index to garentee Id is first column. Because dictionaries are unordered.
#Index is automatically included as 1st column. To exclude ever use to_csv param.
submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predictions2}).set_index('Id')
submission.to_csv('submssions.csv', index=False)


#predictions2.head()

#df = pd.concat([pd.DataFrame(test.Id), pd.DataFrame(predictions)]) #must pass a collection of DFs into concat
#df.head()
#my_submission = pd.DataFrame({'Id': df.Id, 'SalePrice': df.predictions})
# you could use any filename. We choose submission here
#my_submission.to_csv('submission.csv', index=False)