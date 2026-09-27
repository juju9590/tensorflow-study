# 이미지 수치화
from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import matplotlib.pyplot as plt

photo_path = './_data/my_photo/' # 절대경로
my_img = load_img(photo_path + 'yj_3.jpeg', target_size=(150,150),) # load_img는 한장 가져올때 편함                   
               
# print(my_img)
# print(type(my_img)) #<class 'PIL.Image.Image'>
# # <PIL.Image.Image image mode=RGB size=150x150 at 0x2B63A529660>

# plt.imshow(img)
# plt.show()

arr = img_to_array(my_img) # 이미지 수치화 하는 tool
print(arr)
print(arr.shape) #(150, 150, 3) 3차원
print(type(arr)) #<class 'numpy.ndarray'>

# 4차원으로 만들어 주기 위해 reshape
arr = np.expand_dims(arr, axis=0) # 차원증가
print(arr)
print(arr.shape) #(1, 150, 150, 3)

img_path = './_save/my_photo/'
filename = 'yj_3.npy'

np.save(img_path + filename , arr=arr)  # 저장 파일명 
