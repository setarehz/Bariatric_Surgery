from keras.models import Sequential
from keras.layers import Densevenv
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
...


df = pd.read_csv(r'Normalized_April27.csv', encoding = "ISO-8859-1", engine='python')

final_features=['REE','Android Region (%Fat)', 'smoker', 'paintens', 'Total Fat (g)', 'Gynoid Region (%Fat)', 'arthmeds', 'infomeet', 'suprtgrp', 'calcmeds', 'diabmeds','vitdmeds' ]

X = df[final_features]
y = df['Reversal_rate']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=462)

...
# define base model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

# evaluate model
estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)
kfold = KFold(n_splits=10)
results = cross_val_score(estimator, X, Y, cv=kfold)
print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))