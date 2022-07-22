# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 12:56:39 2022

@author: Admin
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


def check_outliers(an_array, max_deviations=2):
    '''
    '''
    #an_array = row[:365].astype(float).values
    mean = np.mean(an_array)
    standard_deviation = np.std(an_array)
    distance_from_mean = abs(an_array - mean)
    
    outliers_indecies = distance_from_mean > max_deviations * standard_deviation
#    no_outliers = an_array[not_outlier]

    return outliers_indecies


#                  imports
# =======================================================

datasets = dict(nsidc_v4='b', jaxa='r', osisaf='g')

idf = pd.read_csv("d:/ml/pr/ice_machine_learning/input_data/X_small_iifp_jon.csv",
                  sep=';', index_col=[0]
                  )
idf['state'] = 1

invalid = pd.read_csv("d:/ml/pr/ice_machine_learning/input_data/X_invalid_jon.csv",
                      sep=';', index_col=[0]
                      )
invalid['state'] = 0

idf = pd.concat((idf, invalid), axis=0)

points = pd.read_csv("d:/ml/pr/ice_machine_learning/preprocessing/meta_309_points.csv", sep=';', index_col=[0])

points['p'] = points.index.values
idf['p'] = idf.index // 41

result = idf.join(points, on='p', how='left',
                  rsuffix="_r",
                  )
result['year'] = 1979 + (result.index % 41)
del result['p_r']


#         
# =======================================================

N = 500   # номер точки в таблице result
t = -1    # внутренний индекс для iloc
treshold = 30   # percents
window = 15
new_X = pd.DataFrame(np.zeros((result.shape[0], 365)))
new_X.columns = ['x{}'.format(s) for s in range(365)]
new_X.index = result.index[:]

target = pd.DataFrame(np.ones((result.shape[0], 365)))
target.columns = ['y{}'.format(s) for s in range(365)]
target.index = result.index[:]


#             adding duration column to result:
# =======================================================

for i, row in result.iterrows():
    
    fig = plt.figure
    t += 1
    
    # if t != N:
    #     continue

    arr = row[:365]
    arr = arr.astype(float)

    i_out = check_outliers(arr.values)
    arr[i_out] = np.nan
#    perc = np.percentile(arr, 0.5)
#    arr[arr < 30] = np.nan

    # arr.plot(zorder=1000, label='no outliers')

    new = arr.interpolate(method='linear', 
                          axis=0, limit_direction='both')

#    new[:15] = arr[:15]
#    new = arr.rolling(window=window).median().values    
#    new[:15] = arr[:15]
    new = 100. * (new - new.min()) / (new.max() - new.min())
    new_X.loc[i, :] = new.values
    
    ii = np.where(new <= treshold)[0]
#    ifp_dur = 
    result.loc[i, 'iifp_Y_dur'] = ii[-1] - ii[0]
    tmp_Y = np.ones(365)
    
    if len(ii) > 1:
        tmp_Y[ii[0] : ii[-1] + 1] = 0
    target.loc[i, :] = tmp_Y

'''
#               combine plot:
# =======================================================

   
    new2 = row[:365].astype(float).rolling(window=window, center=True).median().values
    new2[:window//2] = arr[:window//2]
    new2[window//2:] = arr[window//2:]
    
    
    new2 = 100. * (new2 - new2.min()) / (new2.max() - new2.min())
    jj = np.where(new2 <= treshold)[0]
    tarr = np.ones(365)
    if len(jj) > 1:
        tarr[jj[0] : jj[-1] + 1] = 0
    
    
    arr = np.ones(365)
    if len(ii) > 1:
        arr[ii[0] : ii[-1] + 1] = 0
    
    target.iloc[t, :] = arr
#    result.iloc[t, :365] = new

    
    fig = plt.figure(figsize=(8,6)) 
    plt.plot(100. * tarr, label='target_medF', alpha=0.7, zorder=100)
    result.iloc[t, :365].plot(label='origin', c='k', lw=1., alpha=0.7, zorder=100)
    
    (100. * target.iloc[t, :]).plot(label='target', c='g', alpha=0.7, zorder=100)

    plt.plot(new, label='noF_normed', alpha=0.7, c='purple', zorder=100)

    plt.legend()
    plt.grid(ls=':')
#    plt.savefig(dpi=400)
    # plt.show()
    plt.close()
'''


Y2_jon = pd.read_csv('d:/ml/pr/ice_machine_learning/input_data/Y2_jon.csv', sep=';', index_col=[0])
X_jon = pd.read_csv('d:/ml/pr/ice_machine_learning/input_data/X_jon.csv', sep=';', index_col=[0])

manual_check = pd.read_csv('d:/ml/pr/ice_machine_learning/target_Y/manual_check_iifp_lt_20_or_gt_200.csv', sep=';')
manual_check.rename(columns={'Unnamed: 0':'index'})
manual_check = result.query('iifp_Y_dur < 20 | iifp_Y_dur > 175')


#          ploting manual_check:
# =======================================================

# plt.scatter(manual_check.lon, manual_check.lat)
# plt.grid(ls=':')
# plt.show()


#            Saving 37 plots:
# =======================================================


# for i in manual_check:
#     fig = plt.figure(figsize=(8,6))
#     plt.plot(manual_check.loc[i].iloc[:365], alpha=0.7, zorder=100, color='k', label='')
#     plt.savefig()


new2 = row[:365].astype(float).rolling(window=window, center=True).median().values
new2[:window//2] = arr[:window//2]
new2[window//2:] = arr[window//2:]
    
    
new2 = 100. * (new2 - new2.min()) / (new2.max() - new2.min())
jj = np.where(new2 <= treshold)[0]
tarr = np.ones(365)

if len(jj) > 1:
  tarr[jj[0] : jj[-1] + 1] = 0
  arr = np.ones(365)
if len(ii) > 1:
  arr[ii[0] : ii[-1] + 1] = 0
  target.iloc[t, :] = arr
#    result.iloc[t, :365] = new

fig = plt.figure(figsize=(8,6)) 
plt.plot(100. * tarr, label='target_medF', alpha=0.7, zorder=100)
result.iloc[t, :365].plot(label='origin', c='k', lw=1., alpha=0.7, zorder=100)   
(100. * target.iloc[t, :]).plot(label='target', c='g', alpha=0.7, zorder=100)
plt.plot(new, label='noF_normed', alpha=0.7, c='purple', zorder=100)
plt.legend()
plt.grid(ls=':')
#plt.savefig(dpi=400)
plt.show()
plt.close()

'''
i = 0
while i < 37:
    fig = plt.figure(figsize=(8,6)) 
    plt.plot(manual_check.values[i, :365], label='№{} point_normalize'.format(i+1), alpha=0.7, zorder=100, c='k')
    plt.plot(100. * Y2_jon.values[i, :365], label='target_Y2', alpha=0.7, zorder=100, c='r')
    result.iloc[i, :365].plot(label='origin', c='b', lw=1., alpha=0.7, zorder=100)  
    # plt.plot(100. * tarr.values[i, :], label='target_medF', alpha=0.7, zorder=100, c='m')
    # plt.plot(new, label='noF_normed', alpha=0.7, c='g', zorder=100)
    (100. * target.iloc[i, :]).plot(label='target_t', c='g', alpha=0.7, zorder=100)
    plt.legend()
    plt.grid(ls=':')
    plt.show()
    # plt.savefig('d:/ml/pr/pics/37_pics_manual_target/{}_point_manual.png'.format(i+1), format='png', dpi=350)
    i +=1
'''  
    
#       Delete manual_check indecies from result, X_jon and Y2_jon:
# =======================================================

for i in range(len(manual_check)):
    el = manual_check.index[i]
    result = result.drop(index=[el])
    X_jon = X_jon.drop(index=[el])
    Y2_jon = Y2_jon.drop(index=[el])


#       Result split 70/30:
# =======================================================

result_test, result_train = train_test_split(result, test_size=0.70)  
    
result_train.to_csv('d:/ml/pr/ice_machine_learning/input_data/result_train_70.csv', sep=';')    
result_test.to_csv('d:/ml/pr/ice_machine_learning/input_data/result_test_30.csv', sep=';')


#      Remove 30% result from X_jon and Y2_jon:
# =======================================================   
for i in range(len(result_test)):
    el = result.index[i]
    X3_jon = X_jon.drop(index=[el])
    Y3_jon = Y2_jon.drop(index=[el])


X3_jon.to_csv('d:/ml/pr/ice_machine_learning/input_data/X3_jon.csv', sep=';')
Y3_jon.to_csv('d:/ml/pr/ice_machine_learning/input_data/Y3_jon.csv', sep=';')

