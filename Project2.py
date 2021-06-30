"""
<< 2021 Spring CE553 - Project 2>>

Greenhouse gas emission is thought to cause the earth’s climate to change. Here, we will analyze
the data of carbon dioxide concentration. Calculate a mean and standard deviation of the records
for each of 1971, 1986, 2002, and 2019, and then find a probabilistic distribution describing the
likelihood of the records.
"""
import csv
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


pointOfInterest = ['1971', '1986', '2002', '2019']
rawYearlyCO2, sampleData, populationData = [{} for _ in range(3)]

rawCO2ConcentrationData = open(r'DATA/Project2/CO2Concentration.csv')
for idx, row in enumerate(csv.reader(rawCO2ConcentrationData)):
    try:
        lineItem = [str(row[0]), float(row[1])]
        year = lineItem[0][:4]
        if year in pointOfInterest and lineItem[1] >= 0:  # REMOVE ERROR DATA (-999.99)
            rawYearlyCO2[year] = rawYearlyCO2.get(year, [])
            rawYearlyCO2[year].append(lineItem[1])
    except ValueError:
        print('Value Error in row {0} --> {1}'.format(idx, row), )
rawCO2ConcentrationData.close()

for year in rawYearlyCO2:
    sampleData[year] =\
        {'mean': np.mean(rawYearlyCO2[year]), 'stdev': np.std(rawYearlyCO2[year], ddof=1)}
    populationData[year] =\
        {'mean': np.mean(rawYearlyCO2[year]), 'stdev': np.std(rawYearlyCO2[year])}

    print('Year: {0};\n'
          'Sample mean: {1:.3f} ppm, Sample standard deviation: {2:.4f} ppm'
          .format(year, sampleData[year]['mean'], sampleData[year]['stdev']))
    print('Assume the data follows normal distribution, calculate maximum likelihood estimator.')
    print('Population mean: {0:.3f} ppm, Population standard deviation: {1:.4f} ppm\n'
          .format(populationData[year]['mean'], populationData[year]['stdev']))


colors = ['blue', 'green', 'orange', 'red']
xTicks = [populationData[year]['mean'] for year in populationData]
xTicksLabel = ['$\\mu$ = {0:.2f}\n$\\sigma$ = {1:.4f}'
               .format(populationData[year]['mean'], populationData[year]['stdev'])
               for year in populationData]

figDist = plt.figure(figsize=[7, 5], dpi=300, tight_layout=True)
figDist.suptitle(
    '$\mathregular{CO_{2}}$ Concentration Records and Probabilistic Distribution',
    fontweight='bold')
axisDist = figDist.gca()
axisDist.set_xlabel('$\mathregular{CO_{2}}$ Concentration (ppm)')
axisDist.set_ylabel('p(x)')
axisDist.set_xlim([320, 480])

axisSub = axisDist.twiny()
axisSub.set_xlabel('Yearly $\\mu$ and $\\sigma$ for normal distribution')
axisSub.set_xlim([320, 480])
axisSub.set_xticks(xTicks)
axisSub.set_xticklabels(xTicksLabel)
axisSub.grid(which='major', axis='x')

for idx, year in enumerate(pointOfInterest):
    axisDist.hist(rawYearlyCO2[year],
                  bins=int((max(rawYearlyCO2[year])-min(rawYearlyCO2[year]))),
                  # bins=int(1+3.3*np.log(len(rawYearlyCO2[year]))),
                  density=True, alpha=.3, color=colors[idx], label='Histogram for {0}'.format(year))

    xRange = np.linspace(populationData[year]['mean']-5*populationData[year]['stdev'],
                         populationData[year]['mean']+5*populationData[year]['stdev'],
                         50)
    axisDist.plot(xRange,
                  stats.norm.pdf(xRange, populationData[year]['mean'], populationData[year]['stdev']),
                  color=colors[idx], label=str(year))
axisDist.legend()
figDist.savefig('Result/Project2/CO2Concentration.png')
plt.show()
