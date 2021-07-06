"""
<< 2021 Spring CE553 - Project 8>>

Convolution;
Air quality in a city is being monitored.
We analyze the concentration of fine particulate matter (PM2.5) in the city of Beijing.
Apply a wavelet transform on the time series, and then find a pattern,
such as the peak value and duration of fine-dust attacks, in every 1 year cycle.
Again, discuss the result for the feature extraction.
"""
import csv
import datetime
import pywt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scaleogram as scg
from mpl_toolkits.axes_grid1 import make_axes_locatable


column = ['Index', 'DateTime', 'PM2.5', 'DEWP', 'TEMP', 'PRES', 'CBWD', 'Lws', 'Is', 'Ir']
BeijingPM25Data = open('DATA/Project8/BeijingPM2.5Data.csv', 'r')
BeijingPM25 = []
for idx, row in enumerate(csv.reader(BeijingPM25Data)):
    try:
        BeijingPM25.append(
            [int(row[0]),
             datetime.datetime(*[int(item) for item in row[1:5]]),
             int(row[5]), float(row[6]), float(row[7]), float(row[8]),
             str(row[9]), float(row[10]), float(row[11]), float(row[12])])
    except ValueError:
        print('Value Error in row {0} --> {1}'.format(idx, row))
BeijingPM25Data.close()
BeijingPM25 = pd.DataFrame(BeijingPM25, columns=column)


averageLength = 24*7*4
BeijingPM25['RollAvg'] = BeijingPM25.iloc[:, 2].rolling(window=averageLength).mean()
BeijingPM25.head()

wavelet = 'cmor1-2'
scg.set_default_wavelet(wavelet)
scales = scg.periods2scales(np.arange(1, 24*7*5, 1))
data = BeijingPM25['PM2.5'].values.squeeze()


fig, ax = plt.subplots(2, 1, figsize=(7, 10), dpi=300, tight_layout=True)
fig.suptitle('Beijing PM 2.5 Wavelet Transform with {} Wavelet'.format(wavelet))
ax[0].set_title('PM2.5 of Beijing')
ax[0].plot(BeijingPM25['Index'], BeijingPM25['RollAvg'], color='darkcyan', label=f'{averageLength} hour average')
sc = ax[0].scatter(BeijingPM25['Index'], data, c=data, cmap='magma', s=1)

divider = make_axes_locatable(ax[0])
cax = divider.append_axes('right', size="3%", pad=0.15)
plt.colorbar(sc, cax=cax, label='µg/m^3')

ax[0].legend()
ax[0].set_ylabel('PM2.5 µg/m^3')
ax[0].set_xticks(np.arange(0, 5)*365*24)
ax[0].set_xticklabels(2010 + np.arange(0, 5))

scg.cws(data, scales=scales,
        title='Scalogram', xlabel="Year", ylabel="Hours",
        cmap='magma', coi=False, ax=ax[1],
        coikw={'color': 'black', 'alpha': 0.7, 'hatch': '/'})
ax[1].set_xticks(np.arange(0, 5)*365*24)
ax[1].set_xticklabels(2010 + np.arange(0, 5))
plt.show()
