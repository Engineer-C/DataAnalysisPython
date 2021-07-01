"""
<< 2021 Spring CE553 - Project 1>>

Data type and portability: DailyTemperatureStation, DailyTemperatureDaejeon
The first project is for handling various data and getting familiar with a programming language.
Demonstrate your spatial, categorical, and time-series data by plotting them in an appropriate form.
"""
import csv
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import shapefile as shp
from pyproj import Transformer


tempDataDaejeon = []
quarterMarker = []
month = '-02-'
tempByMonth, minTempByMonth, maxTempByMonth = \
    [[[], [], [], [], [], [], [], [], [], [], [], []] for _ in range(3)]
monthlyAvgTemp, monthlyMinAvgTemp, monthlyMaxAvgTemp = \
    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for _ in range(3)]
monthIndex = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

rawDataDaejeon = open(r'DATA/Project1/DailyTemperatureDaejeon.csv')
for idx, row in enumerate(csv.reader(rawDataDaejeon)):
    try:
        lineItem = [str(row[0]), int(row[1]), float(row[2]), float(row[3]), float(row[4])]
        tempDataDaejeon.append(lineItem)
        if str(row[0])[4:8] in ['-01-', '-05-', '-09-'] and str(row[0])[4:8] != month:
            quarterMarker.append(idx)
        month = str(row[0])[4:8]
        tempByMonth[int(month[1:3])-1].append(lineItem[2])
        minTempByMonth[int(month[1:3])-1].append(lineItem[3])
        maxTempByMonth[int(month[1:3])-1].append(lineItem[4])
    except ValueError:
        print('Value Error (probably header row): ', row)
rawDataDaejeon.close()

date = [row[0] for row in tempDataDaejeon]
temp = [row[2] for row in tempDataDaejeon]
minTemp = [row[3] for row in tempDataDaejeon]
maxTemp = [row[4] for row in tempDataDaejeon]

for i in range(12):
    monthlyAvgTemp[i] = sum(tempByMonth[i]) / len(tempByMonth[i])
    monthlyMinAvgTemp[i] = sum(minTempByMonth[i]) / len(minTempByMonth[i])
    monthlyMaxAvgTemp[i] = sum(maxTempByMonth[i]) / len(maxTempByMonth[i])


figLine = plt.figure(figsize=[10, 5], dpi=300, tight_layout=True)
figLine.suptitle('Daily Temperature of Daejeon', fontweight='bold')
axLine = figLine.gca()
axLine.set_ylim([-20, 40])

axLine.plot(date, temp, linewidth=1, color='black', label='Daily temperature')
axLine.fill_between(date, minTemp, maxTemp, alpha=.5, label='Daily temperature range')

axLine.set_xlabel('Date')
axLine.set_ylabel('Temperature (℃)')
axLine.set_xticks(quarterMarker)
axLine.grid(which='major', axis='x')
axLine.legend(loc=3)
figLine.savefig('Result/Project1/tempDaejeon.png')
plt.show()


figMonthLine = plt.figure(figsize=[7, 5], dpi=300, tight_layout=True)
figMonthLine.suptitle('Monthly Average Temperature of Daejeon', fontweight='bold')
axMonthLine = figMonthLine.gca()

axMonthLine.plot(monthIndex, monthlyAvgTemp, linewidth=1, color='black',
                 linestyle='-', marker='.', label='Average temperature')
axMonthLine.plot(monthIndex, monthlyMinAvgTemp, linewidth=1, color='blue',
                 linestyle='-', marker='.', label='Average of minimum temperature')
axMonthLine.plot(monthIndex, monthlyMaxAvgTemp, linewidth=1, color='red',
                 linestyle='-', marker='.', label='Average of maximum temperature')

axMonthLine.set_xlabel('Month')
axMonthLine.set_ylabel('Temperature (℃)')
axMonthLine.legend()
figMonthLine.savefig('Result/Project1/monthlyAvgTempDaejeon.png')
plt.show()


tempDataStation = []
rawDataStation = open(r'DATA/Project1/DailyTemperatureStation.csv')
mapKorea = shp.Reader(r'DATA/Project1/CTPRVN_202101/TL_SCCO_CTPRVN.shp', encoding='EUC-KR')

for row in csv.reader(rawDataStation):
    try:
        lineItem = [int(row[0]), str(row[1]), str(row[2]), float(row[3]), float(row[4]), float(row[5])]
        tempDataStation.append(lineItem)
    except ValueError:
        print('Value Error (probably header row): ', row)
rawDataStation.close()

provinceCount = {}
for item in tempDataStation:
    provinceCount[item[2].strip()] = provinceCount.get(item[2].strip(), 0) + 1
provinceLabel = provinceCount.keys()
provinceCount = provinceCount.values()

# %% Pie Plot
figPie = plt.figure(figsize=[7, 5], dpi=300, tight_layout=True)
figPie.suptitle('Number of Stations in Province', fontweight='bold')
axPie = figPie.gca()
axPie.pie(provinceCount, labels=provinceLabel,
          autopct=lambda pct: "{:d}".format(int(pct / 100. * sum(provinceCount))), startangle=90)
figPie.savefig('Result/Project1/provinceCount.png')
plt.show()


UTM_K = 'epsg:5178'
GRS80 = 'epsg:4927'
transformer = Transformer.from_crs(GRS80, UTM_K)

latUTMk = [item[3] for item in tempDataStation]
lonUTMk = [item[4] for item in tempDataStation]
alt = [item[5] for item in tempDataStation]
lat, lon = transformer.transform(latUTMk, lonUTMk)

cmap = cm.get_cmap('jet')
norm = Normalize(vmin=min(alt), vmax=300)
colors = cmap(norm(alt))



fig3d = plt.figure(figsize=[6, 5], dpi=300, tight_layout=True)
fig3d.suptitle('Location and Altitude of Station', fontweight='bold')
ax3d = fig3d.gca(projection='3d')
ax3d.view_init(60, -105)

for shape in mapKorea.shapeRecords():
    for i in range(len(shape.shape.parts)):
        i_start = shape.shape.parts[i]
        if i == len(shape.shape.parts)-1:
            i_end = len(shape.shape.points)
        else:
            i_end = shape.shape.parts[i+1]
        x = [i[0] for i in shape.shape.points[i_start:i_end]]
        y = [i[1] for i in shape.shape.points[i_start:i_end]]
        ax3d.plot(x, y, zs=0, zdir='z', linewidth=.5, color='black')
ax3d.bar3d(lon, lat, 0, 4000, 4000, alt, color=colors, zsort='max')

ax3d.set_xlabel('GRS80 X')
ax3d.set_ylabel('GRS80 Y')
ax3d.set_zlabel('Altitude (m)')
ax3d.set_xticks([800000, 1000000, 1200000, 1400000])
ax3d.set_yticks([1600000, 1800000, 2000000])
ax3d.set_zticks([200, 400, 600, 800])
sc = cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = fig3d.colorbar(sc, ticks=[100, 200, 300], label='Altitude (m)')
cbar.ax.set_yticklabels(['100', '200', '300+'])
fig3d.savefig('Result/Project1/3dMap.png')
plt.show()
