"""
<< 2021 Spring CE553 - Project 6>>

Fourier transform;
A structure shakes at its natural frequency when wind or earthquake invokes its vibration.
Applying fast Fourier transform on the vibration signal allows us to figure out the natural frequency.
As a result, the time-series data can be reduced to the peak-frequency features of the structure.
Discuss the effectiveness of this technique in terms of dimensionality reduction.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


dataFile = 'DATA/Project7/5StorySteelStructure.csv'
vibration = pd.read_csv(dataFile, names=['Time', '1F', '2F', '3F', '4F', '5F'], dtype=np.float64)

fig, ax = plt.subplots(5, 2,
                       figsize=[7, 12], dpi=300, tight_layout=True, gridspec_kw={'width_ratios': [3, 2.5]})
fig.suptitle('Frequency Strength and Spectrogram by Floor', fontweight='bold')

for row, floor in enumerate(['5F', '4F', '3F', '2F', '1F']):
    strength = np.fft.fft(list(vibration[floor]))
    strength = abs(strength)
    peaks = find_peaks(strength, height=110, distance=800)
    freq = np.fft.fftfreq(n=18000, d=.01)

    xTick = [item for item in freq[peaks[0]] if item > 0]
    ax[row, 0].set_title(floor, fontweight='bold', loc='right', pad=2)
    ax[row, 0].set(xlim=[0, 50], xlabel='Frequency (Hz)', ylabel='Strength')
    peakAx = ax[row, 0].twiny()
    peakAx.spines['left'].set_visible(False)
    peakAx.set(xlim=[0, 50], xticks=xTick)
    peakAx.set_xticklabels(['{0:.1f}'.format(item) for item in xTick], fontsize='small', rotation=0)
    peakAx.grid(which='major', axis='x', linestyle='--')

    ax[row, 1].set(xticks=[30, 60, 90, 120, 150, 180], xlabel='Time (s)', ylabel='Frequency (Hz)')

    ax[row, 0].plot(freq, strength, color='indigo')
    ax[row, 1].specgram(list(vibration[floor]), Fs=100, cmap='inferno', NFFT=8192)
plt.show()
