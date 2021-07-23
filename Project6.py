"""
<< 2021 Spring CE553 - Project 6>>

Regression;
Heavy metals contaminate the restoration sites of decommissioned nuclear power plants.
Developing a novel absorbent for removing radionuclides from aqueous environments is critical these days.
Conduct a regression to evaluate the adsorption isotherm of cobalt ions on to the hydroxyapatite beads.
"""
import csv
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


CoOH2 = []
CoOH2RawData = open(r'DATA/Project6/AdsorptionCobaltHydroxyapatite.csv')
for idx, row in enumerate(csv.reader(CoOH2RawData)):
    try:
        CoOH2.append([np.float(item) for item in row])
    except ValueError:
        print('Value Error in row {0} --> {1}'.format(idx, row))
CoOH2RawData.close()
CoOH2 = np.array(CoOH2)


def langmuir_isotherm(xdata, qmax, k):
    qe = qmax * k * xdata / (1 + k * xdata)
    return qe


markers = {239: 'o', 303: '*', 313: '^'}
colors = {239: 'darkcyan', 303: 'gold', 313: 'coral'}
plt.figure(dpi=300, tight_layout=True)
plt.suptitle('Adsorption Isotherm of\nCobalt Ions on to the Hydroxyapatite Beads', fontweight='bold')
ax = plt.gca()
ax.set(xlabel='$\mathregular{C_{e}}$ (mg/L)', ylabel='$\mathregular{q_{e}}$ (mg/g)')

for item in np.unique(CoOH2[:, 0]):
    dataByTemp = np.where(CoOH2[:, 0] == item)
    xData, yData = CoOH2[dataByTemp, 1].flatten(), CoOH2[dataByTemp, 2].flatten()
    xRange = np.linspace(0, max(xData), 200, endpoint=True)
    pOpt, pCov = curve_fit(langmuir_isotherm, xData, yData, method='lm')

    residuals = yData - langmuir_isotherm(xData, *pOpt)
    ssRes = np.sum(residuals ** 2)
    ssTot = np.sum((yData - np.mean(yData)) ** 2)
    rSquared = 1 - (ssRes / ssTot)

    ax.scatter(xData, yData,
               s=50, c=colors[item], marker=markers[item], label=str(item)+'K')
    ax.plot(xRange, langmuir_isotherm(xRange, *pOpt),
            color=colors[item], alpha=.8,
            label='$\mathregular{q_{Max}}$='+'{0:.3f}'.format(pOpt[0])
                  +', K={0:.3f}'.format(pOpt[1])
                  +', $\mathregular{R^{2}}$='+'{0:.4f}'.format(rSquared))
ax.plot([], [], linestyle='-', color='black', label='Langmuir Isotherm')
handles, labels = plt.gca().get_legend_handles_labels()
order = [3, 4, 5, 6, 0, 1, 2]
ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='lower right')
plt.show()
