
import warnings

import lightgbm
from lightgbm.sklearn import LGBMRegressor
warnings.filterwarnings("ignore")

import os
from os.path import join

import pandas as pd
import numpy as np

import missingno as msno

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score
import xgboost as xgb
import lightgbm as lgb

import matplotlib.pyplot as plt
import seaborn as sns


os.chdir('/Users/heechankang/projects/pythonworkspace/git_study/fundamental_node/data')

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print('train data dim : {}'.format(train.shape))
print('sub data dim : {}'.format(test.shape))

train.head()

train['date'] = train['date'].apply(lambda x: x[:6]).astype(int)
train.head()

y = train['price']
del train['price']
train

del train['id']

print(train.columns)

test['date'] = test['date'].apply(lambda x: x[:6]).astype(int)

del test['id']
print(test.columns)

y

sns.kdeplot(y)
plt.show()

y = np.log1p(y)
y

sns.kdeplot(y)
plt.show()

train.info()


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def rmse(y_test, y_pred):
    return np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(y_pred)))


from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

random_state=2020

gboost = GradientBoostingRegressor(random_state=random_state)

xgboost = XGBRegressor(random_state = random_state)
lightgbm = LGBMRegressor(random_state=random_state)
rdforest = RandomForestRegressor(random_state=random_state)

models = [gboost, xgboost, lightgbm, rdforest]

gboost.__class__.__name__

def get_scores(models, train, y):
    df = {}

    for model in models:
        model_name = model.__class__.__name__

        X_train, X_test, y_train, y_test = train_test_split(train, y, random_state=random_state, test_size = 0.2)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        df[model_name] = rmse(y_test, y_pred)

        score_df = pd.DataFrame(df, index = ['RMSE']).T.sort_values('RMSE', ascending=False)
    
    return score_df

get_scores(models, train, y)

from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators':[50, 100],
    'max_depth':[1, 10],
}

model = LGBMRegressor(random_state=random_state)

grid_model = GridSearchCV(model, param_grid=param_grid,
                          scoring = 'neg_mean_squared_error',
                          cv=5, verbose = 1, n_jobs = 5)

grid_model.fit(train, y)

grid_model.cv_results_

params = grid_model.cv_results_['params']
params
score = grid_model.cv_results_['mean_test_score']
score

results = pd.DataFrame(params)
results
results['score'] = score
results

results['RMSE'] = np.sqrt(-1 * results['score'])
results

results=results.rename(columns = {'RMSE':'RMSLE'})
results

results.sort_values(by='RMSLE', inplace = True)
results.reset_index(drop=True)

def my_GridSearch(model, train, y, param_grid, verbose=2, n_jobs=5):
    grid_model = GridSearchCV(model, param_grid=param_grid, scoring='neg_mean_squared_error',
                              cv=5, verbose = verbose, n_jobs=n_jobs)

    grid_model.fit(train, y)

    params = grid_model.cv_results_['params']
    score = grid_model.cv_results_['mean_test_score']

    results = pd.DataFrame(params)
    results['score'] = score

    results['RMSLE'] = np.sqrt(-1 * results['score'])
    results = results.sort_values('RMSLE')

    return results

param_grid = {
    'n_estimators' : [50, 100],
    'max_depth' : [1, 10]
}

model = LGBMRegressor(random_state=random_state)
my_GridSearch(model, train, y, param_grid, verbose=2, n_jobs=5)

model = LGBMRegressor(max_depth=10, n_estimators=100, random_state=random_state)
model.fit(train, y)
prediction = model.predict(test)
prediction

prediction = np.expm1(prediction)
prediction

submission = pd.read_csv('sample_submission.csv')
submission.head()

submission['price']=prediction
submission.head()

submission_csv_path = ('submission_{}_RMSLE_{}.csv'.format('lgbm', '0.164399'))
submission.to_csv(submission_csv_path, index = False)
print(submission_csv_path)

def save_submission(model, train, y, test, model_name, rmsle=None):
    model.fit(train, y)
    prediction = model.predict(test)
    prediction = np.expm1(prediction)
    submission = pd.read_csv('sample_submission.csv')
    submission['price'] = prediction
    submission_csv_path = ('submission_{}_RMSLE_{}.csv'.format(model_name, rmsle))
    submission.to_csv(submission_csv_path, index=False)
    print('{} saved'.format(submission_csv_path))

save_submission(model,train,y,test, 'lgbm', rmsle='0.0168')