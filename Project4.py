"""
<< 2021 Spring CE553 - Project 4>>

Concrete is one of the most important materials to constitute a built environment.
Mixing ingredient materials results in a concrete composite, and its mix proportion determines the strength.
Evaluate the contribution (correlation) of each ingredient material on the concrete strength.
"""
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


mixture28Days, columnHeader = [[] for _ in range(2)]

concreteData = open(r'DATA/Project4/ConcreteStrength.csv')
for idx, row in enumerate(csv.reader(concreteData)):
    try:
        lineItem = [float(row[i]) for i in range(len(row))]
        if lineItem[-2] == 28:
            mixture28Days.append(lineItem)
    except ValueError:
        print('Value Error in row {0} --> {1}'.format(idx, row), )
        columnHeader = row
concreteData.close()
mixture28Days = np.asarray(mixture28Days)

numberOfData = np.size(mixture28Days, axis=0)


def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip


corrFig = plt.figure(figsize=[9, 9], dpi=300, tight_layout=True)
corrFig.suptitle('Correlation of Each Ingredient Material on the Concrete Strength', fontweight='bold')
corrAxis = corrFig.subplots(3, 3)
for idx in range(7):
    notNullRow = np.empty((0, 2))
    # corrAxis = corrFig.add_subplot(2, 4, column+1)
    rowNum = [0, 0, 0, 1, 1, 1, 2]
    colNum = [0, 1, 2, 0, 1, 2, 0]
    corrAxis[rowNum[idx], colNum[idx]].set_title(columnHeader[idx][:-35]+' (Comp.'+str(idx+1)+')',
                                                 fontdict={'fontweight':'bold', 'fontsize':10})
    corrAxis[rowNum[idx], colNum[idx]].set_ylim([0, 90])
    corrAxis[rowNum[idx], colNum[idx]].set_xlabel('kg/$\mathregular{m^{3}}$', fontdict={'fontsize':8})
    corrAxis[rowNum[idx], colNum[idx]].set_ylabel('Conc. Strength (MPa)', fontdict={'fontsize':8})

    for row in range(numberOfData):
        if mixture28Days[row, idx] != 0:
            notNullRow = np.append(notNullRow, [[mixture28Days[row, idx], mixture28Days[row, 8]]], axis=0)

    corrAxis[rowNum[idx], colNum[idx]].scatter(notNullRow[:, 0], notNullRow[:, 1], color='black', s=2)

    position = (notNullRow[:, 0].mean(), notNullRow[:, 1].mean())
    covar = np.cov(notNullRow[:, 0], notNullRow[:, 1])
    print(np.corrcoef(notNullRow[:, 0], notNullRow[:, 1]))
    # conf50 = plot_cov_ellipse(covar, position, nstd=.674, ax=corrAxis[rowNum[idx], colNum[idx]],
    #                           alpha=.2, color='blue', label='50%')
    conf95 = plot_cov_ellipse(covar, position, nstd=1.96, ax=corrAxis[rowNum[idx], colNum[idx]],
                              alpha=.2, color='skyblue', label='95%')

corrAxis[2, 1].spines['top'].set_visible(False)
corrAxis[2, 1].spines['bottom'].set_visible(False)
corrAxis[2, 1].spines['left'].set_visible(False)
corrAxis[2, 1].spines['right'].set_visible(False)
corrAxis[2, 1].tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
corrAxis[2, 2].spines['top'].set_visible(False)
corrAxis[2, 2].spines['bottom'].set_visible(False)
corrAxis[2, 2].spines['left'].set_visible(False)
corrAxis[2, 2].spines['right'].set_visible(False)
corrAxis[2, 2].tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
corrFig.legend([conf95], ['95%'], loc='lower right')
corrFig.savefig('Result/Project4/CorrelationOfConcrete.png')
plt.show()
