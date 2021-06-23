"""
<< 2021 Spring CE553 - Final Exam>>

Students need to complete the final project as described below. Your presentation should be no longer than 5 minutes.
Submit your recorded presentation video, with your face and voice, entitled by your student id #
: email to ********@kaist.ac.kr

1. Find your own dataset for your research.
   It is okay to search a dataset in an open-source website
   such as UCI Machine Learning Repository if you do not have your own.
   Describe your own data in terms of its acquisition and the most importantly purpose of the analysis.
2. Students are expected to use more than 2 techniques that you have practiced in this course,
   or any other good techniques, and finally compile a portfolio of the techniques you are using for the final project.
   For example, the data type, representation, and dimensionality reduction are specified in sequence.
3. Provide an appropriate title and correct references for your final project.

    1) Data Type and Probability
    2) Statistical Estimation
    3) Proportion Estimation
    4) Multidimensional Data --> Correlation
    5) Eigen-space --> PCA
    6) Regression --> LESS
    7) Fourier Transform
    8) Convolution --> Wavelet Transform


Current dataset candidate from https://archive.ics.uci.edu/ml/datasets.php

Seoul Bike Sharing Demand Data Set; https://archive.ics.uci.edu/ml/datasets/Seoul+Bike+Sharing+Demand
    Data Set Characteristics: Multivariate
    Attribute Characteristics: Integer, Real
    Associated Tasks: Classification, Regression
    Number of Instances: 8760 (365 * 24)
    Number of Attributes: 14
    Missing Values? N/A
    Area: Computer
    Date Donated: 2020-03-01
    Number of Web Hits: 29639

Data Set Information;
    Currently Rental bikes are introduced in many urban cities for the enhancement of mobility comfort.
    It is important to make the rental bike available and accessible to the public
    at the right time as it lessens the waiting time.
    Eventually, providing the city with a stable supply of rental bikes becomes a major concern.
    The crucial part is the prediction of bike count required at each hour for the stable supply of rental bikes.
    The dataset contains weather information
    (Temperature, Humidity, Wind speed, Visibility, Dew point, Solar radiation, Snowfall, Rainfall),
    the number of bikes rented per hour and date information.

Attribute Information;
    Date : year-month-day
    Rented Bike count - Count of bikes rented at each hour
    Hour - Hour of he day
    Temperature-Temperature in Celsius
    Humidity - %
    Wind speed - m/s
    Visibility - 10m
    Dew point temperature - Celsius
    Solar radiation - MJ/m2
    Rainfall - mm
    Snowfall - cm
    Seasons - Winter, Spring, Summer, Autumn
    Holiday - Holiday/No holiday
    Functional Day - NoFunc(Non Functional Hours), Fun(Functional hours)

Relevant Papers;
[1] Sathishkumar V E, Jangwoo Park, and Yongyun Cho.
        'Using data mining techniques for bike sharing demand prediction in metropolitan city.'
        Computer Communications, Vol.153, pp.353-366, March, 2020
[2] Sathishkumar V E and Yongyun Cho.
        'A rule-based model for Seoul Bike sharing demand prediction using weather data'
        European Journal of Remote Sensing, pp. 1-18, Feb, 2020
"""
import itertools
import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
import xgboost as xgb
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import seaborn as sns
import imageio


seed = 3593
data = pd.read_csv('DATA/FinalExam/SeoulBikeData.csv', header=0)

'''Data Augmentation'''
# Date into datetime
data['Date'] = [datetime.strptime(date, '%d/%m/%Y') for date in data['Date'].values.astype('str')]

# One hot encoding
data = data.join(pd.get_dummies(data['Date'].dt.strftime('%A')))
data = data.join(pd.get_dummies(data['Seasons']))
data = data.drop(labels=['Seasons'], axis=1)

# String to binary data
data['Holiday'] = [0 if item == 'No Holiday' else 1 for item in data['Holiday']]
data['Functioning'] = [0 if item == 'No' else 1 for item in data['Functioning']]

headers = data.columns
print(data.info(verbose=True, memory_usage=True, show_counts=True))


"""
PCA ANALYSIS
"""
pca = PCA(svd_solver='full')
result = pca.fit_transform(X=data[headers[2:]], y=data['Rented Bike Count'])

'''
Create GIF File to Visualize 3D Data
1 Degree Delta, 360 Images.
Disabled to prevent unnecessary load.
'''
# cmap = cm.get_cmap('viridis')
# norm = Normalize(vmin=0, vmax=2000)
# colors = cmap(norm(data['Rented Bike Count'].values))
#
# plt.figure(figsize=[5, 5], dpi=200, tight_layout=False)
# ax = plt.axes(projection='3d')
#
# ax.scatter(result[:, 0], result[:, 1], result[:, 2],
#            c=colors, cmap=cmap, s=1)
#
# ax.set(xlabel='PC1', ylabel='PC2', zlabel='PC3')
# ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
# ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
# ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
#
# # plt.axis('off')
#
# imagesName = []
# for i, angle in enumerate(np.arange(0, 360)):
#     ax.view_init(30, angle)
#     ax.figure.savefig(f'Result/FinalExam/GIFSource/{i:03d}.png')
#     imagesName.append(f'Result/FinalExam/GIFSource/{i:03d}.png')
#
# images = []
# for image in imagesName:
#     images.append(imageio.imread(image))
# imageio.mimsave('Result/FinalExam/PCA.gif', images)


"""
CREATE DEMAND FORECASTING MODEL
"""
'''Test Data Split'''

# Random Day selections
testIndex = np.array([np.arange(start, start+24).tolist()
                      for start in 24 * (7 * np.random.default_rng(seed).choice(range(52), size=14, replace=False)
                                         + np.array([2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1]))]
                     ).reshape((-1,))
train = data.drop(testIndex)
test = data.iloc[testIndex]

# Search Best Fit Parameters
# choice = input('Search? [Y/n]: ')
choice = 'n'
if choice == 'Y' or choice == 'y':
    grid = GridSearchCV(xgb.XGBRegressor(),
                        {'n_estimators': range(90, 120),
                         'gamma': np.linspace(0.1, 0.9, 30),
                         'max_depth': range(4, 15),
                         'random_state': [seed]},
                        cv=5, verbose=10)

    grid = grid.fit(X=np.ascontiguousarray(train[headers[2:]]),
                    y=np.ascontiguousarray(train['Rented Bike Count']))
    model = grid.best_estimator_
else:
    '''
    Current Best Fit Setup
    2021-06-09 01:13:50.465149
    XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                 colsample_bynode=1, colsample_bytree=1, gamma=0.6793103448275862,
                 gpu_id=0, importance_type='gain', interaction_constraints='',
                 learning_rate=0.300000012, max_delta_step=0, max_depth=11,
                 min_child_weight=1, missing=nan, monotone_constraints='()',
                 n_estimators=94, n_jobs=64, num_parallel_tree=1, random_state=3593,
                 reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
                 tree_method='exact', validate_parameters=1, verbosity=None)
    '''
    model = xgb.XGBRegressor(gamma=0.6793103, max_depth=11, n_estimators=94, random_state=seed)

model.fit(X=np.ascontiguousarray(train[headers[2:]]), y=np.ascontiguousarray(train['Rented Bike Count']))
print(model)

prediction = model.predict(np.ascontiguousarray(test[headers[2:]]))


"""
DATA VISUALIZATION
"""
# Color Theme
primary = 'darkcyan'
secondary = 'coral'
tertiary = 'black'
quaternary = 'goldenrod'

'''Plot Full Data'''
plt.figure(figsize=[11, 5], dpi=200, tight_layout=True)
plt.suptitle('Hourly Rented Bike Count', fontsize='x-large', fontweight='bold')
ax = plt.gca()
ax.plot(data['Date'], data['Rented Bike Count'], color=primary, linewidth=1)
xTickLoc = [datetime.strptime(string, '%d/%m/%Y')
            for string in ['01/12/2017', '01/03/2018', '01/06/2018', '01/09/2018', '1/12/2018']]
ax.set(title=f'From {data.iloc[0]["Date"].strftime("%b %d, %Y")} to {data.iloc[-1]["Date"].strftime("%b %d, %Y")}',
       xticks=xTickLoc, ylabel='Count')


'''Plot Day of Week'''
fig, ax = plt.subplots(1, 7, figsize=[12.5, 5], sharey='row', dpi=200, tight_layout=True)
fig.suptitle('Hourly Average Rented Bike Count by Day of Week', fontsize='x-large', fontweight='bold')
for i, dayOfWeek in enumerate(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']):
    dataByDayOfWeek = data.loc[data[dayOfWeek] == 1]
    hourlyAverage = np.empty(24, dtype='float64')
    for hour in range(24):
        hourlyAverage[hour] = dataByDayOfWeek.loc[dataByDayOfWeek['Hour'] == hour]['Rented Bike Count'].mean()
    ax[i].plot(hourlyAverage, color=[primary if _ <= 4 else secondary for _ in range(7)][i])
    ax[i].set(title=dayOfWeek, xlabel='Hour')
ax[0].set(ylabel='Count')


'''Plot Weekday and Weekend'''
fig, ax = plt.subplots(1, 2, figsize=[10, 5], dpi=200, tight_layout=True)
fig.suptitle('Hourly Average Rented Bike Count of Weekday and Weekend', fontsize='x-large', fontweight='bold')
dataWeekday = data.loc[data['Date'].dt.dayofweek <= 4]
hourlyAverageWeekday = np.empty(24, dtype='float64')
for hour in range(24):
    hourlyAverageWeekday[hour] = dataWeekday.loc[dataWeekday['Hour'] == hour]['Rented Bike Count'].mean()
ax[0].plot(hourlyAverageWeekday, color=primary)
ax[0].set(title='Weekday Trend', xlabel='Hour', ylabel='Count')

dataWeekend = data.loc[data['Date'].dt.dayofweek >= 5]
hourlyAverageWeekend = np.empty(24, dtype='float64')
for hour in range(24):
    hourlyAverageWeekend[hour] = dataWeekend.loc[dataWeekend['Hour'] == hour]['Rented Bike Count'].mean()
ax[1].plot(hourlyAverageWeekend, color=secondary)
ax[1].set(title='Weekend Trend', xlabel='Hour', ylabel='Count')


'''Plot Monthly Average'''
dataSeasonal = pd.DataFrame(np.empty((24, 1)), columns=['Hour'])
dataSeasonal['Hour'] = range(24)
seasonalAverage = np.empty(24, dtype='float64')
for dataSplit, typeOfDay in [(dataWeekday, 'Weekday'), (dataWeekend, 'Weekend')]:
    for season in ['Winter', 'Spring', 'Summer', 'Autumn']:
        for hour in range(24):
            seasonalAverage[hour] = dataSplit.loc[(dataSplit['Hour'] == hour)
                                                  & (dataSplit[season] == 1)]['Rented Bike Count'].mean()
        dataSeasonal[season+typeOfDay] = seasonalAverage

fig, ax = plt.subplots(2, 4, figsize=[8, 5], dpi=200, sharey='row', tight_layout=True)
for header, location in zip(dataSeasonal.columns[1:], itertools.product(range(2), range(4))):
    ax[location].plot(dataSeasonal[header], color=[primary, secondary][location[0]])
    ax[location].set(title=header[:-7] + ' ' + header[-7:])


'''Plot Test Data Results'''
fig, ax = plt.subplots(2, 7, figsize=[21, 6], dpi=200, sharex='col', tight_layout=True)
for i, location in enumerate(itertools.product(range(2), range(7))):
    ax[location].set(title=test.iloc[i*24]['Date'].strftime('%b %d, %Y. %a'), xticks=[0, 6, 12, 18, 24])
    ax[location].plot(np.arange(24), test['Rented Bike Count'].values[i*24:i*24+24],
                      color=[primary if _ <= 4 else secondary for _ in range(7)][test.iloc[i*24]['Date'].dayofweek],
                      linestyle='--', alpha=.8)
    ax[location].plot(np.arange(24), prediction[i*24:i*24+24], color=tertiary, label='Prediction', linewidth=2)
plt.legend()


'''Plot Feature Importance'''
featureImportance = pd.Series(index=headers[2:], data=model.feature_importances_).sort_values(ascending=False)

plt.figure(figsize=[8, 5], dpi=200, tight_layout=True)
plt.bar(featureImportance.index, featureImportance.values, color=primary)
plt.yscale('log')
plt.xticks(rotation=45, ha='right')


'''
Plot Correlation
!!!TAKES A LOT OF TIME TO COMPUTE!!!
Disabled to prevent unnecessary load.
'''
# gridPlot = sns.PairGrid(data[headers[1:11]], diag_sharey=False)
# gridPlot.map_upper(sns.kdeplot, color=primary, fill=True)
# gridPlot.map_diag(sns.histplot, color=secondary)
# gridPlot.map_lower(sns.scatterplot, color=primary)

plt.show()
print()
