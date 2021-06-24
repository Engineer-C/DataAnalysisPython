"""
<< 2021 Spring CE553 - Project 3>>

Taxis are one of the most important mobility services in a city. They usually operate through a taxi
dispatch center, using mobile data terminals installed in the vehicles. Find the population
proportion of (1) central-based demand, (2) stand-based demand, or (3) demand on a random street.
In addition, provides its error range with a 95% level of confidence.
"""
import csv
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


centralBasedDemand, standBasedDemand, randomStreetDemand = [[] for _ in range(3)]

rawTaxiData = open(r'DATA/Project3/PortoTaxiDataTraining.csv')
for idx, row in enumerate(csv.reader(rawTaxiData)):
    try:
        if row[1] == 'A':
            centralBasedDemand.append(row)
        elif row[1] == 'B':
            standBasedDemand.append(row)
        elif row[1] == 'C':
            randomStreetDemand.append(row)
        else:
            print('Not defined CALL_TYPE in row {0} --> {1}'.format(idx, row))
    except ValueError:
        print('Value Error in row {0} --> {1}'.format(idx, row), )
rawTaxiData.close()

totalTaxiCalls = len(centralBasedDemand) + len(standBasedDemand) + len(randomStreetDemand)

centralSampleProp = len(centralBasedDemand) / totalTaxiCalls
standSampleProp = len(standBasedDemand) / totalTaxiCalls
randomSampleProp = len(randomStreetDemand) / totalTaxiCalls

levelOfConfidence = .95
criticalValue = (1 - levelOfConfidence) / 2
zValue = stats.norm.ppf(1 - criticalValue)


def range_calculator(sample_proportion, sample_size, z_value):
    lower_range = sample_proportion - z_value * (sample_proportion * (1 - sample_proportion) / sample_size) ** (1 / 2)
    upper_range = sample_proportion + z_value * (sample_proportion * (1 - sample_proportion) / sample_size) ** (1 / 2)
    scale = (sample_proportion - lower_range) / z_value
    return {'lower': lower_range, 'upper': upper_range, 'stdev': scale}


centralPopulationProp = range_calculator(centralSampleProp, totalTaxiCalls, zValue)
standPopulationProp = range_calculator(standSampleProp, totalTaxiCalls, zValue)
randomPopulationProp = range_calculator(randomSampleProp, totalTaxiCalls, zValue)


# %%
xAxisRange1 = [.22, .27]
xAxisRange2 = [.49, .54]
xTicks1 = [centralPopulationProp['lower'], centralSampleProp, centralPopulationProp['upper'],
           randomPopulationProp['lower'], randomSampleProp, randomPopulationProp['upper']]
xTicks2 = [standPopulationProp['lower'], standSampleProp, standPopulationProp['upper']]

figProp, (axis1, space, axis2) = \
    plt.subplots(1, 3, figsize=[9, 5], dpi=450, sharex='col', sharey='row',
                 gridspec_kw={'width_ratios': [8, 1, 8]})
figProp.suptitle('Population Proportion of Taxis by Demand with 95% Level of Confidence', fontweight='bold')

space.patch.set_alpha(0)
space.spines['right'].set_visible(False)
space.spines['left'].set_visible(False)
space.spines['top'].set_linestyle("dashed")
space.spines['bottom'].set_linestyle('dashed')
space.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
space.set_xlabel('Proportion')

axis1.set_xlim(xAxisRange1)
axis1.set_ylim([0, 85])
axis1.set_ylabel('p(x)')
axis1.spines['right'].set_visible(False)
axis1.yaxis.tick_left()

axis1Sub = axis1.twiny()
axis1Sub.spines['right'].set_visible(False)
axis1Sub.set_xlim(xAxisRange1)
axis1Sub.set_xticks(xTicks1)
axis1Sub.set_xticklabels(['{0:.3f}'.format(item) for item in xTicks1], fontsize='small', rotation='vertical')
axis1Sub.grid(which='major', axis='x', linestyle='--')

axis2.set_xlim(xAxisRange2)
axis2.spines['left'].set_visible(False)
axis2.yaxis.tick_right()

axis2Sub = axis2.twiny()
axis2Sub.spines['left'].set_visible(False)
axis2Sub.set_xlim(xAxisRange2)
axis2Sub.set_xticks(xTicks2)
axis2Sub.set_xticklabels(['{0:.3f}'.format(item) for item in xTicks2], fontsize='small', rotation='vertical')
axis2Sub.grid(which='major', axis='x', linestyle='--')

d = .02
kwargs = dict(transform=axis1.transAxes, color='k', clip_on=False)
axis1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
axis1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
kwargs.update(transform=axis2.transAxes)
axis2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
axis2.plot((-d, +d), (-d, +d), **kwargs)

xRangeCentral = np.linspace(centralSampleProp - 4 * centralPopulationProp['stdev'],
                            centralSampleProp + 4 * centralPopulationProp['stdev'],
                            200)
xRangeStand = np.linspace(standSampleProp - 4 * standPopulationProp['stdev'],
                          standSampleProp + 4 * standPopulationProp['stdev'],
                          200)
xRangeRandom = np.linspace(randomSampleProp - 4 * randomPopulationProp['stdev'],
                           randomSampleProp + 4 * randomPopulationProp['stdev'],
                           200)

areaRangeCentral = np.linspace(centralPopulationProp['lower'], centralPopulationProp['upper'], 200)
areaRangeStand = np.linspace(standPopulationProp['lower'], standPopulationProp['upper'], 200)
areaRangeRandom = np.linspace(randomPopulationProp['lower'], randomPopulationProp['upper'], 200)

axis1.fill_between(areaRangeCentral,
                   stats.norm.pdf(areaRangeCentral, centralSampleProp, centralPopulationProp['stdev']),
                   color='darkcyan', alpha=.2)
axis1.fill_between(areaRangeRandom,
                   stats.norm.pdf(areaRangeRandom, randomSampleProp, randomPopulationProp['stdev']),
                   color='orangered', alpha=.2)
axis2.fill_between(areaRangeStand,
                   stats.norm.pdf(areaRangeStand, standSampleProp, standPopulationProp['stdev']),
                   color='orange', alpha=.2)

axis1.plot(xRangeCentral,
           stats.norm.pdf(xRangeCentral, centralSampleProp, centralPopulationProp['stdev']),
           color='darkcyan', label='Central-based Demand')
axis1.plot(xRangeRandom,
           stats.norm.pdf(xRangeRandom, randomSampleProp, randomPopulationProp['stdev']),
           color='orangered', label='Demand on Random Street')
axis2.plot(xRangeStand,
           stats.norm.pdf(xRangeStand, standSampleProp, standPopulationProp['stdev']),
           color='orange', label='Stand-based Demand')

lines_labels = [ax.get_legend_handles_labels() for ax in figProp.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
figProp.legend(lines, labels, loc='center')
figProp.tight_layout()
figProp.subplots_adjust(wspace=0, hspace=0)
figProp.savefig('Result/Project3/PopulationProportion.png')
plt.show()
