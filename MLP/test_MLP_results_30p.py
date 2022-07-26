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
y_file_path = r'../input_data/target_test_30.csv'
x_file_path = r'../input_data/result_test_30.csv'

# Upload ML model & scaler
loaded_model = pickle.load(open(model_name, 'rb'))
loaded_scaler = pickle.load(open('scaler.pkl', 'rb'))

X = pd.read_csv(x_file_path, sep=';', index_col=[0])
Y = pd.read_csv(y_file_path, sep=';', index_col=[0])
Y = Y.astype(int)

#X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
#                                                    test_size=0.50)

X_scaled = loaded_scaler.fit_transform(X.iloc[:, :365])
Y_predicted = loaded_model.predict(X_scaled)

for N in range(0, 700, 100):

    f1_Y = f1_score(Y.iloc[N, :].values,
                    Y_predicted[N, :] )

    
    forecast = 100. * Y_predicted[N, :]
    analysis = 100. * Y.iloc[N, :].values
    origin = X.iloc[N, :365].values
    
    fig = plt.figure(figsize=(8,6))
    plt.plot(origin, label='origin', c='k', alpha=0.7, zorder=100)
    plt.plot(forecast, label='forecast', c='b', alpha=0.7, zorder=100)
    plt.plot(analysis, label='analysis', c='r', lw=2., alpha=0.5)
    plt.grid(ls=':')
    plt.title('{}  F1 Y_test: {}'.format(N, f1_Y)   )
    plt.legend()
    
    plt.show()


#          F1 score:
# =======================================================

F1_total = f1_score(Y.values, Y_predicted,
                    average='samples')
print(f'F1 total: {F1_total:.3f}')




