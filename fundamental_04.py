#####
# 9-1 사이킷런으로 구현해보는 머신러닝
###

# 버전 확인
import sklearn
print(sklearn.__version__)
        # 0.24.1

# 사이킷런 살펴보기
# 생략

# 주요 모듈 - 회귀 모델 실습
import numpy as np
import matplotlib.pyplot as plt
r = np.random.RandomState(10)
x = 10*r.rand(100)
y = 2*x - 3*r.rand(100)
plt.scatter(x, y)

x.shape     # (100, ) - 입력 데이터
y.shape     # (100, ) - 정답 데이터

# 모델 만들기
from sklearn.linear_model import LinearRegression
        # LinearRegression 함수 불러오기
model = LinearRegression()
        # model 인스턴스 생성
model

# 훈련시키기
model.fit(x, y)
        # 에러
type(x)     # numpy.ndarray 임.
X = x.reshape(100, 1)
type(X)     # numpy.ndarray 임.
x.shape
X.shape
        # x, X 가 똑같은거 아닌가? 라고 생각했는데
x
X
        # 를 보면 명확하게 다른걸 알 수 있음.

model.fit(X, y)

x_new = np.linspace(-1, 11, 100)
X_new = x_new.reshape(100, 1)
y_new = model.predict(X_new)

X_ = x_new.reshape(-1, 1)
X_.shape


from sklearn.metrics import mean_squared_error

error = np.sqrt(mean_squared_error(y, y_new))
plt.scatter(x, y, label = 'input data')
plt.plot(X_new, y_new, color = 'red', label = 'regression Line')


# 사이킷런의 주요 모듈 - dataset

from sklearn.datasets import load_wine
data = load_wine()
type(data)      
        # sklearn.utils.Bunch
        # Bunch란? 딕셔너리와 유사한 데이터타입

print(data)
data.keys()
        # dict_keys(['data', 'target', 'frame', 'target_names', 
        # 'DESCR', 'feature_names'])

data.data

data.data.shape     # (178, 13)

data.data.ndim      # 2

data.target

data.target.shape       # (178,)

data.feature_names
        # 'alcohol',
        # 'malic_acid',
        # 'ash',
        # 'alcalinity_of_ash',
        # 'magnesium',
        # 'total_phenols',
        # 'flavanoids',
        # 'nonflavanoid_phenols',
        # 'proanthocyanins',
        # 'color_intensity',
        # 'hue',
        # 'od280/od315_of_diluted_wines',
        # 'proline'

len(data.feature_names)     # 13

data.target_names

data.DESCR


# 사이킷런 데이터셋을 이용한 분류문제 실습

import pandas as pd
pd.DataFrame(data.data, columns = data.feature_names)

x = data.data
y = data.target


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(x, y)
y_pred = model.predict(x)


from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

print(classification_report(y, y_pred))
print('accuracy = ', accuracy_score(y, y_pred))


# 훈련 데이터와 테스트 데이터 직접 분리하기
from sklearn.datasets import load_wine
data = load_wine()
print(data.data.shape)
print(data.target.shape)

X_train = data.data[:142]
X_test = data.data[142:]
print(X_train.shape, X_test.shape)
y_train = data.target[:142]
y_test = data.target[142:]
print(y_train.shape, y_test.shape)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# 정확도 평가
from sklearn.metrics import accuracy_score
print('정답률 = ', accuracy_score(y_test, y_pred))


# train_test_split 으로 분류하기
from sklearn.model_selection import train_test_split
result = train_test_split(x, y, test_size = 0.2, random_state = 42)

type(result)
len(result)

result[0].shape
result[1].shape
result[2].shape
result[3].shape

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, random_state = 42)

    