# 사진 5만장 다 지워놨음. 필요 시 라인 110 확인해서 복습할것


import numpy as np
from PIL import Image

data = np.zeros([32, 32, 3], dtype = np.uint8)
        # uint8 : 음, 양수 구분 없는 정수
image = Image.fromarray(data, 'RGB')
image.show()
        # 검정색 32x32 이미지

data[:, :] = [255, 0, 0]
image = Image.fromarray(data, 'RGB')
image.show()
        # 빨간색 이미지

# 문제 1
data = np.ones([128, 128, 3], dtype = np.uint8) * 255
image = Image.fromarray(data, 'RGB')
image.show()


# 문제 2
from PIL import Image
import os
os.chdir('/Users/heechankang/projects/pythonworkspace/git_study/fundamental_node')
    # 연습용 파일 경로
image_path = './data/pillow_practice.png'
    # 이미지 열기
image = Image.open(image_path)
image.show()
    # 이미지 가로세로 출력
print('width :', image.width)
print('height :', image.height)
    # JPG 파일로 저장하기
new_image_path = './data/pillow_practice.jpg'
image =image.convert('RGB')
image.save(new_image_path)
image.close()


# 문제 3
from PIL import Image
with Image.open(image_path) as im:
    im.resize((100, 200)).show()
    im.save('./data/pillow_practice(resized).png')


# 문제 4
from PIL import Image
with Image.open(image_path) as im:
    im.crop((300, 100, 600, 400)).show()
    im.save('./data/pillow_practice(cropped).png')



import os
import pickle
    # CIFAR-100 데이터 열기
dir_path = './data/cifar-100-python'
train_file_path = os.path.join(dir_path, 'train')

with open(train_file_path, 'rb') as f:
    train = pickle.load(f, encoding = 'bytes')

print(type(train))
# print(train)

train.keys()
    # 키부터 꺼내보기

type(train[b'filenames'])
    # list

train[b'filenames'][0:5]
    # 5개만 꺼내보기

train[b'data'][0:5]
    # data 열어보기

train[b'data'][0].shape
    # 형태 찍어보기

image_data = train[b'data'][0].reshape([32, 32,3], order = 'F') # order 주의??
image = Image.fromarray(image_data)
# Pillow를 사용해서 Numpy 배열을 Image로 만들어보기
image.show()
    # 90도 돌아감. x, y축 뒤집어 줄 필요가 있음.

image_data = image_data.swapaxes(0, 1)
image = Image.fromarray(image_data)
image.show()

import os
import pickle
from PIL import Image
import numpy
from tqdm import tqdm

os.chdir = '/Users/heechankang/projects/pythonworkspace/git_study/fundamental_node'
dir_path = './data/cifar-100-python'
train_file_path = os.path.join(dir_path, 'train')

# image를 저장할 cifar-100-python의 하위 디렉토리(images)를 생성합니다.
images_dir_path = os.path.join(dir_path, 'images')
if not os.path.exists(images_dir_path):
    os.mkdir(images_dir_path) # images 디렉토리 생성

# 32X32의 이미지 파일 50000개를 생성합니다.
with open(train_file_path, 'rb') as f:
    train = pickle.load(f, encoding='bytes')
    for i in tqdm(range(len(train[b'filenames']))):
        filename = train[b'filenames'][i].decode()
        data = train[b'data'][i].reshape([32, 32, 3], order='F')
        image = Image.fromarray(data.swapaxes(0, 1))
        image.save(os.path.join(images_dir_path, filename))



#####
# openCV 돌려보기
###
import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

while(1):
    # Take each frame
    _, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # Threshold the HSV image to get only blue colors
    mask = cv.inRange(hsv, lower_blue, upper_blue)

    # Bitwise - AND mask and original image
    res = cv.bitwise_and(frame, frame, mask = mask)
    cv.imshow('frame', frame)
    cv.imshow('mask', mask)
    cv.imshow('res', res)
    k = cv.waitKey(5) & 0xFF
    if k ==27:
        break

cv.destroyAllWindows()
cap.release()

#####
# OpenCV (2)
###
import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)


#####
# 비슷한 이미지 찾아내기
###
import os
import pickle
import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

# 전처리 시 생성했던 디렉토리
os.chdir('/Users/heechankang/projects/pythonworkspace/git_study/fundamental_node')
dir_path = './data'

def draw_color_histogram_from_image(file_name):
    image_path = os.path.join(dir_path, file_name)
    # 이미지 열기
    img = Image.open(image_path)
    cv_image = cv2.imread(image_path)

    # Image와 Histogram 그리기
    f = plt.figure(figsize = (10, 3))
    im1 = f.add_subplot(1,2,1)
    im1.imshow(img)
    im1.set_title('Image')

    im2 = f.add_subplot(122)
    color = ('b', 'g','r')
    for i, col in enumerate(color):
        # image에서 i번째 채널의 히스토그램을 뽑아서 (0:blue, 1:green, 2:red)   
        histr = cv2.calcHist([cv_image], [i], None, [256], [0,256])
        im2.plot(histr, color = col) # 그래프를 그릴 때 채널과 맞춰서 그리기
    im2.set_title('Histogram')

draw_color_histogram_from_image('pillow_practice(resized).jpg')


####################
import os
import pickle
import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

os.chdir = '/Users/heechankang/projects/pythonworkspace/git_study/fundamental_node'

dir_path = './data'

def get_histogram(image):
    histogram = []

    for i in range(3):
        channel_histogram = cv2.calcHist(images = [image],
                                         channels = [i],
                                         mask = None,
                                         histSize = [4],
                                         # 히스토그램 구간을 4개로 한다.
                                         ranges = [0, 256])
        histogram.append(channel_histogram)
    
    histogram = np.concatenate(histogram)
    histogram = cv2.normalize(histogram, histogram)

    return histogram


# get_histogram() 확인용 코드
filename = train[b'filenames'][0].decode()
file_path = os.path.join(images_dir_path, filename)
image = cv2.imread(file_path)
histogram = get_histogram(image)
histogram


###
import os
import pickle
import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

def build_histogram_db():
    histogram_db = {}

    # 디렉토리에 모아 둔 이미지 파일 전부를 리스트업
    path = images_dir_path
    file_list = os.listdir(images_dir_path)

    for file_name in tqdm(file_list):
        file_path = os.path.join(images_dir_path, file_name)
        image = cv2.imread(file_path)
        histogram = get_histogram(image)
        histogram_db[file_name] = histogram
    return histogram_db

histogram_db = build_histogram_db()
histogram_db['adriatic_s_001807.png']



def get_target_histogram():
    filename = input("이미지 파일명을 입력하세요: ")
    if filename not in histogram_db:
        print('유효하지 않은 이미지 파일명입니다.')
        return None
    return histogram_db[filename]

target_histogram = get_target_histogram()
target_histogram



def search(histogram_db, target_histogram, top_k=5):
    results = {}
    # Calculate similarity distance by comparing histograms.
    for file_name, histogram in tqdm(histogram_db.items()):
        distance = cv2.compareHist(H1=target_histogram,
                                   H2=histogram,
                                   method=cv2.HISTCMP_CHISQR)

        results[file_name] = distance
    results = dict(sorted(results.items(), key=lambda item: item[1])[:top_k])
    return results

result = search(histogram_db, target_histogram)
result



def show_result(result):
    f = plt.figure(figsize = (10, 3))
    for idx, filename in enumerate(result.keys()):
        img_path = os.path.join(images_dir_path, filename)
        im = f.add_subplot(1, len(result), idx+1)
        img = Image.open(img_path)
        im.imshow(img)

show_result(result)




# 마지막
target_histogram = get_target_histogram()
result = search(histogram_db, target_histogram)
show_result(result)