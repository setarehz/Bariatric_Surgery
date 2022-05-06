from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_graphviz
from io import StringIO
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from IPython.display import Image
from pydot import graph_from_dot_data
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import numpy as np


df = pd.read_csv(r'Normalized_April27.csv', encoding = "ISO-8859-1", engine='python')
final_features=['REE','Android Region (%Fat)', 'smoker', 'paintens', 'Total Fat (g)', 'Gynoid Region (%Fat)', 'arthmeds', 'infomeet', 'suprtgrp', 'calcmeds', 'diabmeds','vitdmeds' ]

X = df[final_features]
y = df['Reversal_rate']

train_ratio = 0.75
validation_ratio = 0.15
test_ratio = 0.10

# train is now 75% of the entire data set
# the _junk suffix means that we drop that variable completely
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_ratio)

# test is now 10% of the initial data set
# validation is now 15% of the initial data set
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio))


regressor = DecisionTreeRegressor(random_state=0)

# fit the regressor with X and Y data
regressor.fit(x_train, y_train)

predicted_y= regressor.predict(x_test)

cv = KFold(n_splits=5, random_state=1, shuffle=True)
#MAE CV
scores = cross_val_score(regressor, x_val, y_val,scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# report performance
print('Mean Absolout Error CV and std: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

#MSE CV
scores2 = cross_val_score(regressor, x_val, y_val,scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
# report performance
print('Mean Square Error CV and std: %.3f (%.3f)' % (np.mean(scores2), np.std(scores2)))

#RMSE CV
scores3 = cross_val_score(regressor, x_val, y_val,scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1)
# report performance
print('Root Mean Square Error CV and std: %.3f (%.3f)' % (np.mean(scores3), np.std(scores3)))

print('mean absolute error (MAE) :',metrics.mean_absolute_error(y_test,predicted_y))
#print('mean absolute percentage error (MAPE) :',round(metrics.mean_absolute_percentage_error(y_test,predicted_y)*100))
print('mean square error (MSE) :',metrics.mean_squared_error(y_test,predicted_y))
print('root mean square error (RMSE) :',metrics.mean_squared_error(y_test,predicted_y,squared=False))

def smape(a, f):
    return 1/len(a) * np.sum(2 * np.abs(f-a) / (np.abs(a) + np.abs(f))*100)

print('SMAPE: ',smape(y_test, predicted_y))