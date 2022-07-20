# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


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
    
    


datasets = dict(nsidc_v4='b', jaxa='r', osisaf='g')

idf = pd.read_csv("../input_data/X_small_iifp_jon.csv",
                  sep=';', index_col=[0]
                  )
idf['state'] = 1

invalid = pd.read_csv("../input_data/X_invalid_jon.csv",
                      sep=';', index_col=[0]
                      )
invalid['state'] = 0

idf = pd.concat((idf, invalid), axis=0)

points = pd.read_csv("../preprocessing/meta_309_points.csv", sep=';', index_col=[0])

points['p'] = points.index.values
idf['p'] = idf.index // 41

result = idf.join(points, on='p', how='left',
                  rsuffix="_r",
                  )
result['year'] = 1979 + (result.index % 41)
del result['p_r']

# ------- Rolling and Normalizing  SANDBOX -------
print('A')

N = 500
t = -1
treshold = 30   # percents
window = 15
new_X = pd.DataFrame(np.zeros((result.shape[0], 365)))
new_X.columns = ['x{}'.format(s) for s in range(365)]
new_X.index = result.index[:]

target = pd.DataFrame(np.ones((result.shape[0], 365)))
target.columns = ['y{}'.format(s) for s in range(365)]
target.index = result.index[:]
for i, row in result.iterrows():
    
    fig = plt.figure
    t += 1
#    if t != N:
#        continue
    arr = row[:365]
    arr = arr.astype(float)

    i_out = check_outliers(arr.values)
    arr[i_out] = np.nan
#    perc = np.percentile(arr, 0.5)
#    arr[arr < 30] = np.nan

#    arr.plot(zorder=1000, label='no outliers')

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
        

    """
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
    plt.plot(100. * tarr, label='target_medF')
    result.iloc[t, :365].plot(label='origin', c='k', lw=1.)
    
    (100. * target.iloc[t, :]).plot(label='target', c='g')

    plt.plot(new, label='noF_normed', alpha=0.5, c='purple')

    plt.legend()
    plt.grid(ls=':')
#    plt.savefig(dpi=400)
    #plt.show()
    plt.close()
    """

#X = pd.read_csv('../input_data/X_jon.csv', sep=';', index_col=[0])
Y_target = pd.read_csv('../input_data/Y_jon.csv', sep=';', index_col=[0])
Y_target.loc[target.index, :] = target.values
Y_target.to_csv('../input_data/Y2_jon.csv', sep=';')

manual_check = result.query('iifp_Y_dur < 20 | iifp_Y_dur > 175')
manual_check.to_csv('manual_check_iifp_lt_20_or_gt_200.csv', sep=';')

plt.scatter(manual_check.lon, manual_check.lat)
plt.grid(ls=':')
plt.show()