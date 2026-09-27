# 이미지 수치화 도구 많다.. 그 중 ImageDataGenerator, 다른 강력한 것도 많다
# 지금 우리가 필요한건 이미지를 수치화 하는것이 목적
# 목표 : 

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from keras.preprocessing.image import ImageDataGenerator

print(np.__version__)

# 클래스 : 기능 + 각각의 변수 존재 // 함수는 '기능'만 있따...
# ImageDataGenerator는 추후 데이터 증폭까지 처리 할 수 있다.
train_datagen = ImageDataGenerator(
    # 수치화 파라미터
    rescale=1./255,             # 스케일링 (이미지니깐 255로 나누고 flaot형으로 하기 위해 1. 점 찍기)

    # 데이터 증폭 or 변환 파라미터
    horizontal_flip=True,       # 수평 뒤집기,
    vertical_flip=True,         # 수직 뒤집기,
    width_shift_range=0.1,      # 평형이동
    height_shift_range=0.1,
    rotation_range=5,           # 각도조절(정해진 각도만큼 이미지 회전)
    zoom_range=1.2,             # 확대
    shear_range=0.7,            # 좌표하나를 고정하고 다른 몇개의 좌표를 이동(한마디로 찌부) 
    fill_mode='nearest',        # 근처값으로 채우다
)  
# 트레인 데이터는 학습에 데이터양의 증가를 위해 
test_datagen = ImageDataGenerator(
    rescale=1./255,
)
# 테스트할 이미지는 변환하면 안 됨. 절대 변환하지 않음 => 데이터 조작이 될 수 있기때문에 

# 파일의 경로
# path_train = './_data/image/brain/train/'  #C:\study\_data\image\brain\train
# path_test = './_data/image/brain/test/'

xy_train = train_datagen.flow_from_directory(            # 디렉토리(폴더)로 부터 트레인 데이터를 가져오겠다.
    path_train,                 # 폴더의 경로
    target_size=(100, 100),     # 수집된 이미지 크기를 맞추기 위한 파라미터
    batch_size=10,              # 이터레이터로 잡혀 있다. 배치사이즈가 10일때 표현 (80, 100, 100, 1)(80,) => 8*(10, 100, 100, 1)(80,) 
    class_mode='binary',        # 이진분류
    color_mode='grayscale',      # 흑백
    shuffle=True,
)
# Found 160 images belonging to 2 classes.
xy_test = test_datagen.flow_from_directory(
    path_test,
    target_size=(100, 100),     
    batch_size=10,              
    class_mode='binary',        
    color_mode='grayscale',     
    shuffle=False, # test에서 필요가 없다 
)
# Found 120 images belonging to 2 classes

print(xy_train)
# Iterator 순차적인 리스트 형태 
# print(xy_train.next()) # 이터레이터의 첫번째를 보여줘
# print(xy_train.next()) # 두번째 이터레이터를 출력해줘

# print(xy_train[0]) # 이터레이터의 첫번째 (x, y 합쳐진 통 데이터 형태)
# print(xy_train[1]) # 이터레이터의 두번째
# print(xy_train[2]) # 이터레이터의 세번째

# print(xy_train[0][0]) #첫번째 배치의 X 데이터
# print(xy_train[0][1]) #첫번째 배치의 y 데이터


print(xy_train[0][0].shape) #(10, 100, 100, 1)
print(xy_train[0][1].shape) #(10,)

print(xy_train[15][0].shape) #(10, 100, 100, 1)

# print(xy_train[16][0]) # 여기부터 에러  이유는 트레인 데이터는 160장, 배치는  10 이니깐
# Found 160 images belonging to 2 classes.

print(type(xy_train)) #<class 'keras.preprocessing.image.DirectoryIterator'> 이터레이터 데이터는 연속된 데이터라고 생각하면 됨
print(type(xy_train[0])) #<class 'tuple'> : x,y 그 값을 튜블로 저장, 수정이 안된다.
print(type(xy_train[0][0])) #<class 'numpy.ndarray'>
print(type(xy_train[0][1])) #<class 'numpy.ndarray'>




