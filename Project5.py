"""
<< 2021 Spring CE553 - Project 5>>

Eigenspace;
Accurate longitude, latitude and altitude information are useful to perform
eco-routing, cyclist routes expedition and others.
Here let us use it to understand principal component analysis.
Reduce the dimension of the data from 3 to 2, and discuss its geographical meaning.
"""
import csv
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize


spatialData3D = []
spatialNetwork = open(r'DATA/Project5/3dSpatialNetwork.csv')
for idx, row in enumerate(csv.reader(spatialNetwork)):
    try:
        spatialData3D.append(np.array(row[1:], dtype=float))
    except ValueError:
        print('Value Error in row {0} --> {1}'.format(idx, row))
spatialNetwork.close()
spatialData3D = np.array(spatialData3D)


scaler = StandardScaler()
scaledData = scaler.fit_transform(spatialData3D)
pca = PCA(n_components=3, svd_solver='full')
pca.fit(X=scaledData)

transformed = pca.fit_transform(X=scaledData)
print(pca.explained_variance_ratio_)
print(pca.get_params())


cmap = cm.get_cmap('viridis')
norm = Normalize(vmin=min(scaledData[:, 2]), vmax=max(scaledData[:, 2]))
colors = cmap(norm(scaledData[:, 2]))

plt.figure(dpi=300, tight_layout=True)
plt.suptitle('Reduced Dimension of Data', fontweight='bold')
plt.scatter(transformed[:, 0], transformed[:, 1], c=colors, s=.5)
plt.savefig('Result/Project5/2dView.png')
plt.show()


V = pca.components_
normal = V[2, :]
xx, yy = np.meshgrid(np.linspace(-3, 3, 1000, endpoint=True), np.linspace(-3, 3, 1000, endpoint=True))
zz = (-normal[0] * xx - normal[1] * yy) * 1. / normal[2]
trimmedPlane = (-2 <= zz) & (zz <= 6)

fig3d = plt.figure(figsize=[5, 4.8], dpi=300, tight_layout=True)
fig3d.suptitle('Original 3D Data with 2D PCA Projection Plane', fontweight='bold')
ax3d = fig3d.add_subplot(projection='3d')
ax3d.set(xlim=[-3, 3], ylim=[-3, 3], zlim=[-2, 6], xlabel='Longitude', ylabel='Latitude', zlabel='Altitude')
ax3d.view_init(35, 110)
ax3d.scatter(scaledData[:, 0], scaledData[:, 1], scaledData[:, 2], s=.5, c=colors)
ax3d.plot_trisurf(xx[trimmedPlane].ravel(), yy[trimmedPlane].ravel(), zz[trimmedPlane].ravel(), alpha=.2)
ax3d.quiver(-3.5, 0, 0, 7, 0, 0, color='k', arrow_length_ratio=0.05)  # x-axis
ax3d.quiver(0, -3.5, 0, 0, 7, 0, color='k', arrow_length_ratio=0.05)  # y-axis
ax3d.quiver(0, 0, -2.5, 0, 0, 9, color='k', arrow_length_ratio=0.05)  # z-axis
plt.savefig('Result/Project5/3dView.png')
plt.show()
