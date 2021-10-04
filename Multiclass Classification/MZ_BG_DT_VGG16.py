import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Flatten, Dropout, BatchNormalization
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
import matplotlib.pyplot as plt

# Image_size = [110,110]
# train_path = 'G:/1510815/ML and DL/Data/MZ_BG_DT_Train'
# valid_path = 'G:/1510815/ML and DL/Data/MZ_BG_DT_Validation'
# img_height = 110
# img_with = 110
# batch_size = 10


# datagen = ImageDataGenerator(rescale=1./255)
# train_generator = datagen.flow_from_directory(directory=train_path,
#                                               target_size=(img_height,img_with),
#                                               classes=['BG','DT','MZ'],
#                                               class_mode='categorical',
#                                               batch_size=batch_size)
# validation_generator = datagen.flow_from_directory(directory=valid_path,
#                                                    target_size=(img_height,img_with),
#                                                    classes=['BG','DT','MZ'],
#                                                    class_mode='categorical',
#                                                    batch_size=batch_size)
# #  Preprocessing 
# vgg = VGG16(input_shape=(img_height,img_with,3), weights='imagenet', include_top=False)
# # Không train các weigths có sẵn
# for layer in vgg.layers:
#     layer.trainable = False
# # Số lượng class
# # folders = glob('G:/1510815/ML and DL/Data/MZ_BG_DT_Train/*')
# # Layers
# model = Sequential()
# model.add(vgg)
# model.add(Flatten())
# model.add(Dense(128,activation='relu'))
# model.add(Dropout(0.2))
# model.add(BatchNormalization())
# model.add(Dense(1,activation="sigmoid"))
# model.compile(optimizer="adam",
#               loss=[keras.losses.CategoricalCrossentropy(from_logits=True),],
#               metrics=["accuracy"],
#               )
# H = model.fit(train_generator,validation_data=validation_generator, epochs=10,verbose=2)
# model.save('G:/1510815/ML and DL/Data/VGG16_first_model')
model=keras.models.load_model('G:/1510815/ML and DL/Data/VGG16_first_model')
# Load 3 ảnh trong tập test để dự đoán
img0 = image.load_img('G:/1510815/ML and DL/Data/BMD/Dtrrs 148.png')
img0  = image.img_to_array(img0)
img0 = img0/255.0
img0  = np.expand_dims(img0, axis=0)
img0_class=np.argmax(model.predict(img0),axis=1)
# img1 = image.load_img('G:/1510815/ML and DL/Data/BMD/Dtrrs 148.png')
# img1 = image.img_to_array(img1)
# img1= img0/255.0
# img1 = np.expand_dims(img1, axis=0)
# img2 = image.load_img('G:/1510815/ML and DL/Data/BMD/Mzlri 151.png')
# img2 = image.img_to_array(img2)
# img2 = img0/255.0
# img2 = np.expand_dims(img2, axis=0)
# img = np.vstack([img0,img1,img2])
# img_class=model.predict_classes(img,batch_size=10) 
# for things in img_class:  
    # if(things == [0]):
    #     print('Bill Gate')
    # elif(things == [1]): 
    #     print('Donald Trump')
    # else:
    #     print('Mark Zuckerberg')
        
print(img0_class)   