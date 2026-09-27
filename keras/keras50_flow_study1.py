from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
import numpy as np 
import matplotlib.pyplot as plt  

# 이미지 불러오기
img_path = './_data/my_photo/'
my_img = load_img(img_path + 'tori_111.jpeg', 
                  target_size=(150,150),
                  )

# 이미지 수치화 하기
img_arr = img_to_array(my_img)
print(img_arr.shape) #(150, 150, 3)

# 3차원 이미지 -> 4차원으로 변환
img_arr = np.expand_dims(img_arr, axis=0)
print(img_arr.shape) #(1, 150, 150, 3)

# 저장하기 
save_path = './_save/my_photo/'
np.save(save_path +'tori_111.jpeg', arr = img_arr)


datagen = ImageDataGenerator(
    rescale = 1./255,       

    # 요기부터 증폭
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.3,
    rotation_range=5,
    zoom_range=0.1,
    shear_range=0.7,
    fill_mode='nearest'
)

it = datagen.flow(img_arr,
             batch_size=1,  # 이미지 1장이니 통 배치
             )


# print(it)
# <keras.src.legacy.preprocessing.image.NumpyArrayIterator object at 0x000001D04FED9E50>
# print(next(it))  # 파이썬 3.11부터 바뀜 
print(next(it).shape)  # (1, 150, 150, 3)

fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(5,5)) # 1행 5열

for i in range(5) :  #i= 0~4 
    batch = next(it)  #it=변환된 1장의 사진을 batch로 변환 (4차원)
    batch = batch.reshape(150,150,3)  # 4차원의 batch를 3차원으로 변환
    ax[i].imshow(batch) # ax[0] 부터 ax[4] 까지 총 5장의 사진을 꺼내 보여준다
    # ax[i].axis('off')

plt.show()





