
import os
import pickle

import numpy
import matplotlib.pyplot as plt

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

model_name = "sic_MLP_100L.sav"
scaler_name = "scaler.pkl"
y_file_path = r"../target_Y/Y_invalid_jon.csv"
x_file_path = r"../target_Y/X_invalid_jon.csv"

# Upload ML model & scaler
loaded_model = pickle.load(open(model_name, 'rb'))
loaded_scaler = pickle.load(open("scaler.pkl", 'rb'))

X = pd.read_csv(x_file_path, sep=';', index_col=[0])
Y = pd.read_csv(y_file_path, sep=';', index_col=[0])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size=0.50)
X_test_scaled = loaded_scaler.fit_transform(X_test.iloc[:, :365])
Y_test_predicted = loaded_model.predict(X_test_scaled)

N = 50
forecast = 100. * Y_test_predicted[N, :]
analysis = 100. * Y_test.iloc[N, :].values
origin = X_test.iloc[N, :365].values

plt.plot(origin, label='origin', c='k')
plt.plot(forecast, label='forecast', c='b')
plt.plot(analysis, label='analysis', c='r', lw=2., alpha=0.5)
plt.grid(ls=':')
plt.legend()

plt.show()



