# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 15:12:25 2022

@author: Admin
"""

import os
import pickle

import numpy
import matplotlib.pyplot as plt

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

model_name = 'sic_MLP_100L.sav'
scaler_name = 'scaler.pkl'
y_file_path = r'd:/ml/pr/ice_machine_learning/input_data/Y3_jon.csv'
x_file_path = r'd:/ml/pr/ice_machine_learning/input_data/X3_jon.csv'

# Upload ML model & scaler
loaded_model = pickle.load(open(model_name, 'rb'))
loaded_scaler = pickle.load(open('scaler.pkl', 'rb'))

X = pd.read_csv(x_file_path, sep=';', index_col=[0])
Y = pd.read_csv(y_file_path, sep=';', index_col=[0])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size=0.50)

X_test_scaled = loaded_scaler.fit_transform(X_test.iloc[:, :365])
Y_test_predicted = loaded_model.predict(X_test_scaled)

N = 100
forecast = 100. * Y_test_predicted[N, :]
analysis = 100. * Y_test.iloc[N, :].values
origin = X_test.iloc[N, :365].values

fig = plt.figure(figsize=(8,6))
plt.plot(origin, label='origin', c='k', alpha=0.7, zorder=100)
plt.plot(forecast, label='forecast', c='b', alpha=0.7, zorder=100)
plt.plot(analysis, label='analysis', c='r', lw=2., alpha=0.5)
plt.grid(ls=':')
plt.legend()

plt.show()


#        Testing on 30% result data (result_test): 
# =======================================================

result_test = pd.read_csv('d:/ml/pr/ice_machine_learning/input_data/Y3_jon.csv', sep=';', index_col=[0])

result_test_scaled = loaded_scaler.fit_transform(result_test.iloc[:, :365])
Y_result_predicted = loaded_model.predict(result_test_scaled)


N = 13
forecast = 100. * Y_result_predicted[N, :]
analysis = 100. * Y_test.iloc[N, :].values
origin = result_test.iloc[N, :365].values

fig = plt.figure(figsize=(8,6))
plt.plot(origin, label='origin', c='k', zorder=100)
plt.plot(forecast, label='forecast', c='b', zorder=100)
plt.plot(analysis, label='analysis', c='r', lw=2., alpha=0.5)
plt.grid(ls=':')
plt.legend()

plt.show()


#          F1 score:
# =======================================================

f1_Y = f1_score(Y_test, Y_test_predicted, average='samples')
f1_R = f1_score(result_test, Y_result_predicted, average='samples')

print('\nF1 Y_test: ', f1_Y)   
print('\nF1 result_test: ', f1_R)





