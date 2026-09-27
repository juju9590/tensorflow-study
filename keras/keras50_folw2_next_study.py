from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.datasets import fashion_mnist

import numpy as np
import matplotlib.pyplot as plt  

(x_train, y_train),(x_test, y_test) = fashion_mnist.load_data()

datagen = ImageDataGenerator(
    rescale = 1./255,

    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    shear_range=0.7,
    zoom_range=0.1,
    fill_mode='nearest',
)

argment_size = 100 # 증가할 사이즈는  100장

print(x_train.shape) #(60000, 28, 28)
print(x_train[0].shape) #(28, 28)  # 6만장 중 첫번째 장
print(x_train[15].shape) #(28, 28)  # 6만장 중 16번째 장

aaa = np.tile(x_train[0], argment_size) #첫번째 사진을 100장 붙이다.
print(aaa.shape) # (28, 2800) : 옆으로 100장을 붙인 상태

bbb = np.tile(x_train[0], argment_size).reshape(-1,28,28,1)
print(bbb.shape) #(100, 28, 28, 1) 100장을 쌓아 놓은것 => 4차원으로 변환

xy_data = next(datagen.flow(
    np.tile(x_train[0], argment_size).reshape(-1,28,28,1),
    np.zeros(argment_size),
    batch_size=argment_size,
    shuffle=False,
))

print(len(xy_data))  # 2 => 왜냐하면 x, y 2개니깐
print(xy_data[0].shape) #(100, 28, 28, 1)
print(xy_data[1].shape) #(100,)


plt.figure(figsize=(10,10))
for i in range(100):
    plt.subplot(10,10,i+1)
    plt.imshow(xy_data[0][i],cmap='gray')
plt.show()





