import numpy as np 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt 
from keras.preprocessing import image
img_height = 110
img_with = 110
batch_size = 10
model = keras.Sequential([
    layers.Input((110,110,3)),
    layers.Conv2D(16,3,padding='same'),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Flatten(), 
    layers.Dense(3)])

ds_train=tf.keras.preprocessing.image_dataset_from_directory(
    'G:/1510815/ML and DL/Data/MZ_BG_DT_Train',
    labels="inferred",
    label_mode="int",
    class_names=['BG','DT','MZ'],
    color_mode="rgb",
    batch_size=batch_size,
    image_size=(img_height, img_with),
    shuffle=True,
    seed=123,
    validation_split=0.1,
    subset="training",    
)
ds_validation=tf.keras.preprocessing.image_dataset_from_directory(
    'G:/1510815/ML and DL/Data/MZ_BG_DT_Train',
    labels="inferred",
    label_mode="int",
    class_names=['BG','DT','MZ'],
    color_mode="rgb",
    batch_size=batch_size,
    image_size=(img_height, img_with),
    shuffle=True,
    seed=123,
    validation_split=0.1,
    subset="validation",    
)
ds_test=tf.keras.preprocessing.image_dataset_from_directory(
    'G:/1510815/ML and DL/Data/MZ_BG_DT_Test',
    labels="inferred",
    label_mode="int",
    class_names=['BG','DT','MZ'],
    color_mode="rgb",
    
    image_size=(img_height, img_with),
         
)

def augment(x,y):
    image = tf.image.random_brightness(x,max_delta=0.05)
    return image,y
ds_train = ds_train.map(augment)
   
model.compile(
    optimizer = keras.optimizers.Adam(),
    loss = [
        keras.losses.SparseCategoricalCrossentropy(from_logits=True),],
    metrics=["accuracy"],
    )

H = model.fit(ds_train,validation_data=ds_validation, epochs=10,verbose=2)

model.save('G:/1510815/ML and DL/Data/CNN_first_model')
model.save_weights('G:/1510815/ML and DL/Data/weight_CNN_first_model.h5')

# Trực quan hóa
fig = plt.figure()
numofEpoch = 10
plt.plot(np.arange(0, numofEpoch),H.history['loss'],label='training loss')
plt.plot(np.arange(0, numofEpoch),H.history['val_loss'],label='validation loss')
plt.plot(np.arange(0, numofEpoch),H.history['accuracy'],label='accuracy')
plt.plot(np.arange(0, numofEpoch),H.history['val_accuracy'],label='validation accuracy')
plt.title('Accuracy and Loss') 
plt.xlabel('Epoch') 
plt.ylabel('Loss|Accuracy') 
plt.legend()

# Đánh giá model bằng tập test
score = model.evaluate(ds_test, verbose=0)
print(score)

# Load 3 ảnh trong tập test để dự đoán
img0 = image.load_img('G:/1510815/ML and DL/Data/BMD/Bglri 155.png')
img0  = image.img_to_array(img0)
img0  = np.expand_dims(img0, axis=0)
img1 = image.load_img('G:/1510815/ML and DL/Data/BMD/Mzlri 151.png')
img1 = image.img_to_array(img1)
img1 = np.expand_dims(img1, axis=0)
img2 = image.load_img('G:/1510815/ML and DL/Data/BMD/Dtrrs 148.png')
img2 = image.img_to_array(img2)
img2 = np.expand_dims(img2, axis=0)
img = np.vstack([img0,img1,img2])
img_class=model.predict_classes(img,batch_size=10) 
print(img_class)
for things in img_class:  
    if(things == [0]):
        print('Bill Gate')
    elif(things == [1]): 
        print('Donald Trump')
    else:
        print('Mark Zuckerberg')
        
    

