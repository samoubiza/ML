from skimage.io import imread , imshow, show
from skimage import img_as_float
from sklearn.cluster import KMeans
import numpy as np
import math
image = imread('C:/temp/machine learning/courseraYa/parrots.jpg')
#print(len(image[0]))
image_array = img_as_float(image)
nRows, nCols, n = image.shape

##print(image_array)
r = np.array([image_array[:,:,0].ravel()]).T
g = np.array([image_array[:,:,1].ravel()]).T
b = np.array([image_array[:,:,2].ravel()]).T
##print(r)
##print(g)
##print(b)
##
result = np.hstack((r,g))
result = np.hstack((result,b))
##print('result')
##print(len(result))
#n_clusters=8
kmeans_model = KMeans(n_clusters=10, init='k-means++' , random_state=241).fit(result)
##
##
cluster_idx = kmeans_model.labels_
cluster_center = kmeans_model.cluster_centers_
pixel_labels = cluster_idx.reshape(nRows,nCols)
#print(pixel_labels)
##
##res2 = res.reshape((img.shape))
##print(len(cluster_idx))
##print(cluster_idx[:200])
##print(cluster_center)

image_clusters = np.copy(image_array)
for i in range(nRows):
    for j in range(nCols):
        image_clusters[i,j] = cluster_center[pixel_labels[i,j]]

##print(image_clusters[1,712])
mse = np.mean((image_array - image_clusters) ** 2)
psnr = 10 * math.log10(float(1) / mse)
print(psnr)



imshow(image_clusters)
show()