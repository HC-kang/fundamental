from sklearn.datasets import make_blobs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

points, labels = make_blobs(n_samples = 100, centers = 5, n_features = 2, random_state = 135)

print(points.shape, points[:10])
print(labels.shape, labels[:10])

fig = plt.figure()
ax = fig.add_subplot(111)

points_df = pd.DataFrame(points, columns = ['X', 'Y'])
display(points_df.head())

ax.scatter(points[:, 0], points[:, 1], c = 'black', label = 'random generated data')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legent()
ax.grid()

from sklearn.cluster import KMeans

kmeans_cluster = KMeans(n_clusters = 5)

kmeans_cluster.fit(points)


print(type(kmeans_cluster.labels_))
print(np.shape(kmeans_cluster.labels_))
print(np.unique(kmeans_cluster.labels_))


color_dict = {0: 'red', 1: 'blue', 2:'green', 3: 'brown', 4: 'indigo'}

fig = plt.figure()
ax = fig.add_subplot(111)

for cluster in range(5):
    cluster_sub_points = points[kmeans_cluster.labels_==cluster] 
    ax.scatter(cluster_sub_points[:, 0], cluster_sub_points[:, 1], c = color_dict[cluster], label = 'cluster_{}'.format(cluster))

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()
ax.grid()


# 잘 맞지 않는 경우들
from sklearn.datasets import make_circles

circle_points, circle_labels = make_circles(n_samples = 100, factor = 0.5, noise = 0.01)

fig = plt.figure()
ax = fig.add_subplot(111)

circle_kmeans = KMeans(n_clusters=2)
circle_kmeans.fit(circle_points)
color_dict = {0: 'red', 1: 'blue'}
for cluster in range(2):
    cluster_sub_points = circle_points[circle_kmeans.labels_ == cluster]
    ax.scatter(cluster_sub_points[:, 0], cluster_sub_points[:, 1], c=color_dict[cluster], label='cluster_{}'.format(cluster))
ax.set_title('K-means on circle data, K=2')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()
ax.grid()


# 잘 맞지 않는 경우2
from sklearn.datasets import make_moons

moon_points, moon_labels = make_moons(n_samples=100, noise=0.01)

fig = plt.figure()
ax = fig.add_subplot(111)

moon_kmeans = KMeans(n_clusters = 2)
moon_kmeans.fit(moon_points)
color_dict = {0:'red',1:'blue'}
for cluster in range(2):
    cluster_sub_points = moon_points[moon_kmeans.labels_ == cluster]
    ax.scatter(cluster_sub_points[:, 0], cluster_sub_points[:, 1], c=color_dict[cluster], label='cluster_{}'.format(cluster))
ax.set_title('K-means on moon-shaped data, K=2')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()
ax.grid()


# 잘 맞지 않는 경우
from sklearn.datasets import make_circles, make_moons, make_blobs

diag_points, _ = make_blobs(n_samples = 100, random_state=170)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
diag_points = np.dot(diag_points, transformation)

fig = plt.figure()
ax = fig.add_subplot(111)

diag_kmeans = KMeans(n_clusters=3)
diag_kmeans.fit(diag_points)
color_dict = {0:'red', 1:'blue', 2:'green'}
for cluster in range(3):
    cluster_sub_points = diag_points[diag_kmeans.labels_==cluster]
    ax.scatter(cluster_sub_points[:, 0], cluster_sub_points[:,1], c = color_dict[cluster], label = 'cluster_{}'.format(cluster))
ax.set_title('K-means on diagonal-shaped data, K=2')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()
ax.grid()



# DBSCAN
from sklearn.cluster import DBSCAN

fig = plt.figure()
ax = fig.add_subplot(111)
color_dict = {0: 'red', 1: 'blue', 2:'green', 3: 'brown', 4: 'purple'}

epsilon, minPts = 0.2, 3
circle_dbscan = DBSCAN(eps = epsilon, min_samples= minPts)
circle_dbscan.fit(circle_points)
n_cluster = max(circle_dbscan.labels_) + 1

print(f'# of cluster : {n_cluster}')
print(f'DBSCAN Y-hat : {circle_dbscan.labels_}')

for cluster in range(n_cluster):
    cluster_sub_points = circle_points[circle_dbscan.labels_ == cluster]
    ax.scatter(cluster_sub_points[:, 0], cluster_sub_points[:, 1], c=color_dict[cluster], label='cluster_{}'.format(cluster))
ax.set_title('DBSCAN on circle data')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()
ax.grid()


# 달 모양
fig = plt.figure()
ax = fig.add_subplot(111)
color_dict = {0: 'red', 1: 'blue', 2:'green', 3: 'brown', 4: 'purple'}

epsilon, minPts = 0.4, 3
moon_dbscan = DBSCAN(eps = epsilon, min_samples=minPts)
moon_dbscan.fit(moon_points)
n_cluster = max(moon_dbscan.labels_)+1

print(f'# of cluster : {n_cluster}')
print(f'DBSCAN Y-hat : {moon_dbscan.labels_}')

for cluster in range(n_cluster):
    cluster_sub_points = moon_points[moon_dbscan.labels_==cluster]
    ax.scatter(cluster_sub_points[:,0], cluster_sub_points[:,1], c = color_dict[cluster], label='cluster_{}'.format(cluster))
ax.set_title('DBSCAN on moon data')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()
ax.grid()



# 대각 모양
fig = plt.figure()
ax = fig.add_subplot(111)
color_dict = {0: 'red', 1: 'blue', 2:'green', 3: 'brown', 4: 'purple'}

epsilon, minPts = 0.7, 3
diag_dbscan = DBSCAN(eps = epsilon, min_samples=minPts)
diag_dbscan.fit(diag_points)
n_cluster = max(diag_dbscan.labels_)+1

print(f'# of cluster : {n_cluster}')
print(f'DBSCAN Y-hat : {diag_dbscan.labels_}')

for cluster in range(n_cluster):
    cluster_sub_points = diag_points[diag_dbscan.labels_==cluster]
    ax.scatter(cluster_sub_points[:,0], cluster_sub_points[:,1], c = color_dict[cluster], label='cluster_{}'.format(cluster))
ax.set_title('DBSCAN on diag data')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()
ax.grid()

#####
# DBSCAN 알고리즘과 K-means 알고리즘의 소요 시간 비교
###

import time
n_samples = [100, 500, 1000, 2000, 5000, 7400, 10000, 20000, 30000, 40000, 50000]

kmeans_time = []
dbscan_time = []
x = []
for n_sample in n_samples:
    dummy_circle, dummy_labels = make_circles(n_samples = n_sample, factor=0.5, noise=0.01) # 원형 분포 데이터 생성

    kmeans_start = time.time()
    circle_kmeans = KMeans(n_clusters=2)
    circle_kmeans.fit(dummy_circle)
    kmeans_end = time.time()

    dbscan_start = time.time()
    epsilon, minPts = 0.2, 3
    circle_dbscan = DBSCAN(eps = epsilon, min_samples=minPts)
    circle_dbscan.fit(dummy_circle)
    dbscan_end = time.time()
    x.append(n_sample)
    kmeans_time.append(kmeans_end-kmeans_start)
    dbscan_time.append(dbscan_end-dbscan_end)
    print('# of samples : {} / Elapsed time of K-means : {:.5f}s / DBSCAN : {:.5f}s'.format(n_sample, kmeans_end-kmeans_start, dbscan_end-dbscan_start))
    
# K-means와 DBSCAN의 소요 시간 그래프화