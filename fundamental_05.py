from sklearn.datasets import make_blobs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

points, labels = make_blobs(n_samples = 100, centers = 5, n_features = 2, random_state = 135)
        # 분포 만들기

print(points.shape, points[:10])
print(labels.shape, labels[:10])
        # 형태, 예시 찍어보기

fig = plt.figure()
ax = fig.add_subplot(111)
        # 빈박스 만들기

points_df = pd.DataFrame(points, columns = ['X', 'Y'])
display(points_df.head())
        # 위에서 만든 points로 DataFrame 만들기

ax.scatter(points[:, 0], points[:, 1], c = 'black', label = 'random generated data')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()
ax.grid()
        # 검정색 점으로 시각화




#####
# KMeans 해보기
###
from sklearn.cluster import KMeans

kmeans_cluster = KMeans(n_clusters = 5)
        # kmeans_cluster 라는 이름으로 인스턴스 생성
        # n_cluster : 클러스터 개수 지정, 기본값은 8
kmeans_cluster.fit(points)
        # points 를 가지고 훈련시키기 - 비지도 학습이라 input에 labels가 없음!

print(type(kmeans_cluster.labels_))
print(np.shape(kmeans_cluster.labels_))
print(np.unique(kmeans_cluster.labels_))

color_dict = {0: 'red', 1: 'blue', 2:'green', 3: 'brown', 4: 'indigo'}

fig = plt.figure()
ax = fig.add_subplot(111)

for cluster in range(5):
    cluster_sub_points = points[kmeans_cluster.labels_==cluster] 
    ax.scatter(cluster_sub_points[:, 0], cluster_sub_points[:, 1], c = color_dict[cluster], label = 'cluster_{}'.format(cluster))
        # points를 군집화한 labels 를 기준으로 각 cluster_sub_points에 저장 후
        # scatter로 뿌려줌.

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()
ax.grid()



##### 
# 잘 맞지 않는 경우 1 - 원형
###
from sklearn.datasets import make_circles

circle_points, circle_labels = make_circles(n_samples = 100, factor = 0.5, noise = 0)
        # 도넛 모양으로 생성, factor는 내외부 원의 크기 비율.

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
ax.legend(loc = 'best')
ax.grid()



##### 
# 잘 맞지 않는 경우 1 - 달 모양
###
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



##### 
# 잘 맞지 않는 경우 1 - 대각선형
###
from sklearn.datasets import make_circles, make_moons, make_blobs

diag_points, _ = make_blobs(n_samples = 100, random_state=170)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
diag_points = np.dot(diag_points, transformation)
        # 행렬곱으로 blob을 대각선으로 비틀기

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



##### 
# DBSCAN 활용해보기
###
from sklearn.cluster import DBSCAN

fig = plt.figure()
ax = fig.add_subplot(111)
color_dict = {0: 'red', 1: 'blue', 2:'green', 3: 'brown', 4: 'purple'}

epsilon, minPts = 0.2, 3
        # 튜플형식으로 값 한번에 지정해주기.
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

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(x, kmeans_time, c = 'red', marker = 'x', label = 'K-means elapsed time')
ax.scatter(x, dbscan_time, c = 'green', label = 'DBSCAN elapsed time')
ax.set_xlabel('# of samples')
ax.set_ylabel('time(s)')
ax.legend()
ax.grid()




#####
# 차원축소(1) - PCA
###
# 1. 데이터 개요
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
        # 차원축소 예제 - 유방암 데이터셋

cancer = load_breast_cancer()
        # 데이터 불러오기

        # y = 0(Malignant - 악성 종양), y = 1(Benign - 양성 종양)
cancer_X, cancer_y = cancer.data, cancer['target']
train_X , test_X, train_y, test_y = train_test_split(cancer_X, cancer_y, test_size = 0.1, random_state = 10)
        # 데이터 분할
print('전체 검사자 수 : {}'.format(len(cancer_X)))
print('Train dataset에 사용되는 검사자 수 : {}'.format(len(train_X)))
print('Test Dataset에 사용되는 검사자 수 : {}'.format(len(test_X)))
cancer_df = pd.DataFrame(cancer_X, columns = cancer['feature_names'])
cancer_df.head()
cancer_df.info()
cancer_df.describe()

# 2. 유방암 데이터셋에 PCA 알고리즘 적용 예제
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.metrics import accuracy_score
from collections import Counter

# color dict
color_dict = {0:'red', 1:'blue', 2:'red', 3:'blue'}
target_dict = {0:'marlignant_train', 1:'benign_train', 2:'mailgnant_test', 3:'benign_test'}

# Train data에 PCA 알고리즘 적용
train_X_ = StandardScaler().fit_transform(train_X)
        # 불러온 데이터에 대한 정규화 -> 각 column의 range of value가 전부 다르기에 정규화 진행
train_df = pd.DataFrame(train_X_, columns = cancer['feature_names'])
pca = PCA(n_components=2)
        # 주성분 수 2개, 즉 기저 방향벡터를 2개로 하는 PCA 알고리즘 시행
pc = pca.fit_transform(train_df)
        # 정규화 하는 이유 : 예를 들어, mean radius와 mean texture의 값이 같다면 
        # 같은 영향력을 끼칠텐데 두가지의 범위가 다름. 즉 중요도가 왜곡될 수 있으므로 0~1사이로 정규화 실시.

# Test data에 PCA 알고리즘 적용
test_X = StandardScaler().fit_transform(test_X) 
test_df = pd.DataFrame(test_X, columns = cancer['feature_names'])
pca_test = PCA(n_components = 2)
pc_test = pca_test.fit_transform(test_df)


# 훈련한 classifier의 decision boundary를 그리는 함수
def plot_decision_boundary(X, clf, ax):
    h = .02 # step size in the mesh
    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contour(xx, yy, Z, cmap = 'Blues')

# PCA를 적용한 train data의 classifier 훈련 : classifier 로 SVM을 사용한다는 정도만 알아둡시다.
clf = svm.SVC(kernel = 'rbf', gamma = 0.5, C = 0.8) # 여기서는 classifier로 SVM을 사용한다
clf.fit(pc, train_y) # train data로 classifier훈련

# PCA를 적용하지 않은 original data의 SVM 훈련
clf_orig = svm.SVC(kernel = 'rbf', gamma = 0.5, C = 0.8)
clf_orig.fit(train_df, train_y)

# 캔버스 도식
fig = plt.figure()
ax = fig.add_subplot(111)

# malignant와 benign의 SVM decision boundary 그리기
plot_decision_boundary(pc, clf, ax)

# Train data 도식
for cluster in range(2):
    sub_cancer_points = pc[train_y == cluster]
    ax.scatter(sub_cancer_points[:,0], sub_cancer_points[:, 1], edgecolor = color_dict[cluster], c = 'none', label = target_dict[cluster])
# Test data 도식
for cluster in range(2):
    sub_cancer_points = pc_test[test_y == cluster ]
    ax.scatter(sub_cancer_points[:, 0], sub_cancer_points[:, 1], marker = 'x', c = color_dict[cluster+2], label = target_dict[cluster+2])
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('PCA-Breast cancer dataset')
ax.legend()
ax.grid()

# Scoring
pca_test_accuracy_dict = Counter(clf.predict(pc_test) == test_y)
pca_test_accuracy_dict
orig_test_accuracy_dict = Counter(clf_orig.predict(test_df) == test_y)
orig_test_accuracy_dict



print("PCA 분석을 사용한 Test dataset accuracy: {}명/{}명\n\
         => {:.3f}".format(pca_test_accuracy_dict[True],sum(pca_test_accuracy_dict.values()),
         (pca_test_accuracy_dict[True] / sum(pca_test_accuracy_dict.values()))))
print("PCA를 적용하지 않은 Test dataset accuracy: {}명/{}명\n\
         => {:.3f}".format(orig_test_accuracy_dict[True], sum(orig_test_accuracy_dict.values()),
         orig_test_accuracy_dict[True] / sum(orig_test_accuracy_dict.values())))



#####
# T-SNE
### 
print('실행중입니다... 시간이 다소 걸릴 수 있어요. :) \n===')
from sklearn.datasets import fetch_openml

# 784px로 이루어진 mnist 이미지 데이터 호출
mnist = fetch_openml('mnist_784', version = 1)

X = mnist.data / 255.0
y = mnist.target
print('X shape : ', X.shape)
print('Y shape : ', y.shape)

n_image = X.shape[0]
n_image_pixel = X.shape[1]

pixel_columns = [f'pixel{i}' for i in range(n_image_pixel) ] # 픽셀 정보가 있는 columns의 이름을 담은 목록
len(pixel_columns)

import pandas as pd
df = pd.DataFrame(X, columns = pixel_columns)
df
df['y'] = y
df['label'] = df['y'].apply(lambda i: str(i)) # 숫자 라벨을 스트링으로 만드는 람다 함수를 시행
X, y = None, None

import numpy as np

# 결과 고정을 위해 seed 설정
np.random.seed(30)

# 이미지와 데이터 순서를 랜덤으로 뒤바꾼 배열 저장
rndperm = np.random.permutation(n_image)
rndperm
# df['pixel0'] = 0.0
# df

# 랜덤으로 섞은 이미지 중 10,000개를 뽑아 df_subset에 담기
n_image_sample = 10000
random_idx = rndperm[:n_image_sample]
df_subset = df.loc[rndperm[:n_image_sample],:].copy()
df_subset.shape


import seaborn as sns
import matplotlib.pyplot as plt

plt.gray()
fig = plt.figure(figsize = (10, 6))
n_img_sample = 15
width, height = 28, 28

# 15개 샘플 시각화
for i in range(0, n_img_sample):
    row = df_subset.iloc[i]
    ax = fig.add_subplot(3, 5, i+1, title = f'Digit : {row["label"]}')
    ax.matshow(row[pixel_columns]
               .values.reshape((width, height))
               .astype(float))
plt.show()



#####
# PCA를 이용한 MNIST 차원축소
###
from sklearn.decomposition import _pca
print('df_subset의 shape : {}'.format(df_subset.shape))

n_dimension = 2 # 축소시킬 목표 차원 수
pca = PCA(n_components = n_dimension)

df_subset['pixel0'] = 0.0
df_subset

pca_result = pca.fit_transform(df_subset[pixel_columns].values) # 차원 축소 결과
df_subset['pca-one'] = pca_result[:,0] # 축소한결과의 첫번째 차원값
df_subset['pca-two'] = pca_result[:,1] # 축소한 결과의 두번째 차원값

print('pca_result의 shape : {}'.format(pca_result.shape))

print(f'pca-1: {round(pca.explained_variance_ratio_[0], 3)*100}%')
print(f'pca-2: {round(pca.explained_variance_ratio_[1], 3)*100}%')

plt.figure(figsize = (10, 6))
sns.scatterplot(
        x='pca-one', y='pca-two',
        hue='y',
        palette = sns.color_palette('hls', 10),
        data = df_subset,  # 2개의 PC축만 남은 데이터프레임 dt_subset 시각화
        legend = 'full',
        alpha = 0.4
)


#####
# T_SNE 를 이용한 MNIST 차원 축소
###
from sklearn.manifold import TSNE

print('df_subset 의 shape : {}'.format(df_subset.shape))

data_subset = df_subset[pixel_columns].values
n_dimension = 2
tsne = TSNE(n_components = n_dimension)
tsne_results = tsne.fit_transform(data_subset)

print('tsne_results의 shape : {}'.format(tsne_results.shape))

# tsne결과를 차원별로 추가
df_subset['tsne-2d-one'] = tsne_results[:,0]
df_subset['tsne-2d-two'] = tsne_results[:,1]

# 시각화 해보기
plt.figure(figsize = (10, 6))
sns.scatterplot(
        x = 'tsne-2d-one', y = 'tsne-2d-two',
        hue = 'y',
        palette = sns.color_palette('hls', 10),
        data = df_subset,
        legend = 'full',
        alpha = 0.3
)

