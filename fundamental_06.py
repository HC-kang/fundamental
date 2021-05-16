# import requests
# import os
# os.chdir('/Users/heechankang/projects/pythonworkspace/git_study/fundamental_node')
# os.mkdir('anomaly_detection/kospi')

# url = "https://query1.finance.yahoo.com/v7/finance/download/%5EKS11?period1=867715200&period2=1597276800&interval=1d&events=history"
# response = requests.get(url)

# csv_file = './anomaly_detection/kospi/kospi.csv'

# # response의 컨텐츠를 csv로 저장
# with open(csv_file, 'w') as fp:
#     fp.write(response.text)

######## 원본파일 수정으로 인해 사용 불가 -> 새로운 파일 다운로드 및 배치 완료
csv_file = './anomaly_detection/kospi/kospi.csv'

import pandas as pd
df = pd.read_csv(csv_file)
df.head(2)
df.info()
df.describe()

# 날짜 데이터를 Datetime 형식으로 바꿔주기
df.loc[:, 'Date'] = pd.to_datetime(df.Date)

# 데이터의 정합성을 확인
df.isna().sum()

print('삭제 전 데이터 길이(일자 수) :', len(df))
df = df.dropna(axis = 0).reset_index(drop = True)

print('삭제 후 데이터 길이(일자 수) :', len(df))
df.isna().sum()

# 그래프 그려서 확인하기
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams

plt.rcParams['figure.figsize'] = (10, 5)
# LineGraph by matplotlib with wide-form DataFrame

plt.plot(df.Date, df.Close, marker = 's', color = 'r')
plt.plot(df.Date, df.High, marker = 'o', color = 'g')
plt.plot(df.Date, df.Low, marker = '*', color = 'b')
plt.plot(df.Date, df.Open, marker = '+', color = 'y')

plt.title('KOSPI', fontsize = 20)
plt.ylabel('Stock', fontsize = 14)
plt.xlabel('Date', fontsize = 14)
plt.legend(['Close', 'High', 'Low', 'Open'], fontsize = 12, loc = 'best')
plt.show()

df.loc[df.Low > df.High]
df.loc[df.Date == '2020-05-06', 'Low'] = 1902.555078
df.loc[df.Low > df.High]


### 정규분포 그리기
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
x = np.arange(-5, 5, 0.001)
y = norm.pdf(x, 0, 1)
# 평균이 0이고, 표준편차가 1인 정규분포

# build the plot
fig, ax = plt.subplots(figsize=(9, 6))
ax.fill_between(x, y, 0, alpha = 0.3, color = 'b')
ax.set_xlim([-4, 4])
ax.set_title('normal distribution')
plt.show()

fig, ax = plt.subplots(figsize = (9, 6))
_ = plt.hist(df.Close, 100, density = True, alpha = 0.75)

from statsmodels.stats.weightstats import ztest
_, p = ztest(df.Close)
print(p)

#####
# Time series decomposition
###
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(df.Close, model = 'additive', two_sided=True,
                            period = 50, extrapolate_trend = 'freq')
                            # 계절적 성부 50일로 가정
result.plot()
plt.show()

# 그래프 확대
fig, axes = plt.subplots(ncols = 1, nrows = 4, sharex = True, figsize = (12,9))
result.observed.plot(ax = axes[0], legend = False)
axes[0].set_ylabel('Observed')
result.trend.plot(ax = axes[1], legend = False)
axes[1].set_ylabel('Trend')
result.seasonal.plot(ax = axes[2], legend = False)
axes[2].set_ylabel('Seasonal')
result.resid.plot(ax=axes[3], legend = False)
axes[3].set_ylabel('Residual')
plt.show()


# seasonal 다시 확인
result.seasonal[:100].plot()

fig, ax = plt.subplots(figsize = (9, 6))
_ = plt.hist(result.resid, 100, density= True, alpha = 0.75)

r = result.resid.values
st, p = ztest(r)
print(st, p)


#####
# 3sigma 기준 신뢰구간으로 이상치 찾기
###
# 평균과 표준편차 출력
mu, std = result.resid.mean(), result.resid.std()
print('평균 :', mu, '표준편차 :', std)

# 3-sigma 를 기준으로 이상치 판단
print('이상치 갯수 :', len(result.resid[(result.resid>mu + 3*std)|(result.resid<mu-3*std)]))



#####
# 클러스터링으로 이상치 찾기
###
# 데이터 전처리
def my_decompose(df, features, freq = 50):
    trend = pd.DataFrame()
    seasonal = pd.DataFrame()
    resid = pd.DataFrame()

    # 사용할 feature 마다 decompose를 수행
    for f in features:
        result = seasonal_decompose(df[f], model = 'additive', period = freq, extrapolate_trend=freq)
        trend[f] = result.trend.values
        seasonal[f] = result.seasonal.values
        resid[f] = result.resid.values
    
    return trend, seasonal, resid

# 각 변수별 트렌드 / 계절적 / 잔차
tdf, sdf, rdf = my_decompose(df, features = ['Open', 'High', 'Low', 'Close', 'Volume'])
tdf.describe()
rdf.describe()

# 표준 정규화
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(rdf)
print(scaler.mean_)
norm_rdf = scaler.transform(rdf)
norm_rdf

#####
# K-means로 이상치 탐색하기
###
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 2, random_state = 0).fit(norm_rdf)
print(kmeans.labels_)

# 라벨은 몇 번 그룹인지 뜻합니다.
# return_counts = True를 해서 개수까지 확인
lbl, cnt = np.unique(kmeans.labels_, return_counts = True)
print(lbl) # 0, 1번 그룹으로 분할
print(cnt)

# 다시 분류
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 15, random_state = 0).fit(norm_rdf)
lbl, cnt = np.unique(kmeans.labels_,return_counts = True,)
['group:{} - count:{}'.format(group, count) for group, count in zip(lbl, cnt)]


# 분류 결과에서 특이 그룹으로 분류된 그룹 번호로 바꿔주기
df[(kmeans.labels_==2)|(kmeans.labels_==7)|(kmeans.labels_==11)]

df.describe()

# 2004-04-14 주변 정황
df.iloc[1660:1670]


### 각 그룹은 어떤 특징을 가졌는지?
pd.DataFrame(kmeans.cluster_centers_, columns = ['Open', 'High', 'Low', 'Close', 'Volume'])
df.describe()


fig = plt.figure(figsize = (15, 9))
ax = fig.add_subplot(111)
df.Close.plot(ax=ax, label = 'Observed', legend=True, color='b')
tdf.Close.plot(ax=ax, label = 'Trend', legend = True, color = 'r')
rdf.Close.plot(ax = ax, label = 'Resid', legend = True, color = 'y')
plt.show()


from sklearn.cluster import DBSCAN
clustering = DBSCAN(eps = 0.7, min_samples = 2).fit(norm_rdf)
clustering

# 분류된 라벨 확인
print(clustering.labels_)

lbl, cnt = np.unique(clustering.labels_, return_counts = True)
['group:{}-count:{}'.format(group, count) for group, count in zip(lbl, cnt)]


#####
# Auto encoder
###
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

tf.random.set_seed(777)
np.random.seed(777)

### LSTM을 이용한 오토인코더 모델 만들기
# 데이터 전처리
from sklearn.preprocessing import StandardScaler

# 데이터 전처리 - 하이퍼 파라미터
window_size = 10
batch_size = 32
features = ['Open', 'High', 'Low', 'Close', 'Volume']
n_features = len(features)
TRAIN_SIZE = int(len(df)*0.7)

# 데이터 전처리
# 표준정규분포화합니다.

scaler = StandardScaler()
scaler = scaler.fit(df.loc[:TRAIN_SIZE, features].values)
scaled = scaler.transform(df[features].values)

# Keras TimeseriesGenerator 를 이용한 데이터셋 만들기
train_gen = TimeseriesGenerator(
    data = scaled,
    targets = scaled,
    length = window_size,
    stride = 1,
    sampling_rate = 1,
    batch_size = batch_size,
    shuffle = False,
    start_index = 0,
    end_index = None,
)

valid_gen = TimeseriesGenerator(
    data = scaled,
    targets = scaled,
    length = window_size,
    stride = 1, 
    sampling_rate = 1,
    batch_size = batch_size,
    shuffle = False,
    start_index = TRAIN_SIZE,
    end_index = None,
)

print(train_gen[0][0].shape)
print(train_gen[0][1].shape)


# 모델 만들기
model = Sequential([
    # >> 인코더 시작
    LSTM(64, activation = 'relu', return_sequences = True,
         input_shape = (window_size, n_features)),
    LSTM(16, activation = 'relu', return_sequences = False),
    ## << 인코더 끝
    ## >> Bottleneck
    RepeatVector(window_size),
    ## << Bottleneck
    ## >> 디코더 시작
    LSTM(16, activation = 'relu', return_sequences = True),
    LSTM(64, activation = 'relu', return_sequences = False),
    Dense(n_features)
    ## << 디코더 끝
])

model.summary()

# 체크포인트
# 학습을 진행하며 validation 결과가 가장 좋은 모델을 저장해둠.
import os

checkpoint_path = '/Users/heechankang/projects/pythonworkspace/git_study/fundamental_node/anomaly_detection/kospi/mymodel.ckpt'
checkpoint = ModelCheckpoint(checkpoint_path,
                             save_weights_only = True,
                             save_best_only = True,
                             monitor='val_loss',
                             verbose = 1)
# 얼리스탑
# 학습을 진행하며 validation 결과가 나빠지면 스톱. patience 횟수만큼은 참고 지켜본다
early_stop = EarlyStopping(monitor = 'val_loss', patience = 5)
model.compile(loss = 'mae', optimizer = 'adam', metrics = 'mae')
hist = model.fit(train_gen,
                 validation_data = valid_gen,
                 steps_per_epoch = len(train_gen),
                 validation_steps = len(valid_gen),
                 epochs = 50,
                 callbacks = [checkpoint, early_stop])

model.load_weights(checkpoint_path)

# 그래프로 확인
fig = plt.figure(figsize = (12, 8))
plt.plot(hist.history['loss'], label = 'Training')
plt.plot(hist.history['val_loss'], label = 'Validation')
plt.legend()

# 예측 결과를 pred로, 실적 데이터를 real로 받습니다.
pred = model.predict(train_gen)
real = scaled[window_size:]

mae_loss = np.mean(np.abs(pred-real), axis = 1)

# 샘플 개수가 많으니 y축을 로그 스케일로 그리기
fig, ax = plt.subplots(figsize=(9, 6))
_ = plt.hist(mae_loss, 100, density=True, alpha = 0.75, log = True)




import copy
test_df = copy.deepcopy(df.loc[window_size:]).reset_index(drop = True)
test_df['Loss'] = mae_loss

threshold = 3
test_df.loc[test_df.Loss > threshold]


threshold = 0.3
test_df.loc[test_df.Loss > threshold]

# 그래프로 다시 확인
fig = plt.figure(figsize = (12, 15))

#가격들 그래프
ax = fig.add_subplot(311)
ax.set_title('Open/Close')
plt.plot(test_df.Date, test_df.Close, linewidth = 0.5, alpha = 0.75, label = 'Close')
plt.plot(test_df.Date, test_df.Open, linewidth = 0.5, alpha = 0.75, label = 'Open')
plt.plot(test_df.Date, test_df.Close, 'or', markevery = [mae_loss > threshold])

# 거래량 그래프
ax = fig.add_subplot(312)
ax.set_title('Volume')
plt.plot(test_df.Date, test_df.Volume, linewidth = 0.5, alpha = 0.75, label = 'Volume')
plt.plot(test_df.Date, test_df.Volume, 'or', markevery = [mae_loss > threshold])

# 오차율 그래프
ax = fig.add_subplot(313)
ax.set_title('Loss')
plt.plot(test_df.Date, test_df.Loss, linewidth = 0.5, alpha = 0.75, label = 'Loss')
plt.plot(test_df.Date, test_df.Loss, 'or', markevery = [mae_loss > threshold])