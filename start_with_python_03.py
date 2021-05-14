import time
start = time.time()

a = 1
for i in range(10000):
    a+=1

print("time :", time.time() - start)

##############################

my_list = ['a', 'b','c','d','e']

for i in my_list:
    print("값 : ", i)

################################
# enumerate

for i, value in enumerate(my_list):
    print('순번 : ', i,'값 : ', value)

################################
#2중 for문

my_list = ['a', 'b', 'c', 'd']
result_list = []

for i in range(2):
    for j in my_list:
        result_list.append((i,j))

print(result_list)

################################
# 리스트 컴프리헨션

my_list = ['a' , 'b', 'c', 'd']
result_list = [[i, j] for i in range(2) for j in my_list]
print(result_list)

###################################
# 제너레이터
import time
my_list = ['a', 'b', 'c', 'd']
start = time.time()

# 인자로 받은 리스트를 가공해서 만든 데이터셋 리스트을 리턴하는 함수
def get_dataset_list(my_list):
    result_list = []
    for i in range(20000):
        for j in my_list:
            result_list.append((i, j))
    print('>> {} data loaded..'.format(len(result_list)))
    return result_list

for X, y in get_dataset_list(my_list):
    print(X, y)


print("time :", time.time() - start)

#########

import time
my_list = ['a', 'b', 'c', 'd']
start = time.time()

# 인자로 받은 리스트로부터 데이터를 하나씩 가져오는 제너레이터를 리턴하는 함수
def get_dataset_generator(my_list):
    result_list = []
    for i in range(20000):
        for j in my_list:
            yield(i, j)
            print('>> 1 data loaded..')

dataset_generator = get_dataset_generator(my_list)
for X, y in dataset_generator:
    print(X, y)

print("time :", time.time() - start)


#######################

a = 10
b = 0

try:
    # 실행 코드
    print(a/b)

except:
    # 에러가 발생했을 때 처리하는 코드
    print('에러가 발생했습니다.')

################################

a = 10
b = 0

try:
    print(a/b)
except:
    print('에러가 발생했습니다.')
    # 에러가 발생했을 때 처리하는 코드
    b = b+1
    print('값 수정 : ', a/b)

#############################
# 순차처리
import time
num_list = ['p1', 'p2', 'p3', 'p4']
start=time.time()

def count(name):
    for i in range(0, 100000000):
        a=1+2

    print('finish : ', name)

for num in num_list:
    count(num)
print('time :', time.time()-start)
############################
# 병렬처리
import multiprocessing
import time

num_list = ['p1', 'p2', 'p3', 'p4']
start = time.time()

def count(name):
    for i in range(0, 100000000):
        a = 1+2

    print('finish : ', name)

if __name__ == '__main__':
    pool=multiprocessing.Pool(processes = 4)
    pool.map(count, num_list)
    pool.close()
    pool.join()

print("time : ", time.time() - start)