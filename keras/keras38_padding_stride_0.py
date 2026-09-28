import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten

from tensorflow.keras.datasets import mnist

#2. 모델구성
model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(10,10,1), # 9,9,10
                #  padding='valid', # 디폴트 :  패딩을 적용하지 말라
                 padding='same', # 빈공간을 0으로 채워주기 # 10,10,10
                #  strides=1, # 디폴트 : 1칸 움직인다. 소실나지 않은 범위에서 중복하는게 Cnn에서는 먹힌다.
                # 레이어상의 구조를 맞출때 패딩을 많이 사용한다.
                # 가장자리의 특징을 잡아낼때 사용
                 strides=2, #2*2 커널사이즈에서는 사용 지양 , 데이터의 소실로 주의 필요
# strides=1 디폴트
# padding 은 쉐이프 유지, 맥스 풀링 같은경우 남은 부분 짤라버리는데 패딩을 입혀 쉐이프 유지할때 사용한다.

                 ))

model.add(Conv2D(filters=9, kernel_size=(3,3), # 7,7,9
                 padding='valid',
                #  strides=1,
                 strides=2,# 짜투리 남는거 날려버린다. 왜? Shape를 그대로 유지해야하는 상황이기 때문에


                 ))

model.summary()