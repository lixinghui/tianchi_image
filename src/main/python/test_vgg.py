import pandas as pd
import numpy as np
import os
import cv2
import tensorflow as tf
from keras.layers import *
from keras.models import *
from keras.callbacks import *
from keras.preprocessing.image import *
from keras.applications import ResNet50,VGG16
from keras.optimizers import Adam
import matplotlib.pyplot as plt

##设置模型参数
batch_size=16
steps_per_epoch=6
epochs=1

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)    


## 导入vgg模型，并固定所有卷积层
model_vgg=VGG16(include_top=False,input_shape=(224,224,3),pooling=None)
for layer in model_vgg.layers[:-4]:
    layer.trainable=False

## 构造模型并进行编译
X=model_vgg.output
X=Flatten()(X)
X=Dense(128,activation="relu")(X)
Y=Dense(1,activation="sigmoid")(X)
model=Model(inputs=model_vgg.input,outputs=Y)

model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["acc"])
model.summary()

## 设置data_augmentation
gen1=ImageDataGenerator(
    zca_whitening=False,
    zca_epsilon=1e-6,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.,
    zoom_range=0.2,
    channel_shift_range=0.,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=True,
    vertical_flip=True,)
gen2=ImageDataGenerator()

train_gen=gen1.flow_from_directory("../../../data/xls/train",batch_size=batch_size,target_size=(224,224),
class_mode="binary",shuffle=True)

val_gen=gen2.flow_from_directory("../../../data/xls/val",target_size=(224, 224),
        batch_size=batch_size,class_mode='binary')

## 设置学习率递减
reduce_lr=ReduceLROnPlateau(monitor='loss',
factor=0.2,
patience=5,
verbose=1,
min_lr=1e-5)

history=model.fit_generator(train_gen,steps_per_epoch=steps_per_epoch,epochs=epochs,
callbacks=[reduce_lr],validation_data=val_gen,validation_steps=1)
files=[]
predictions=[]
for file in os.listdir("../../../data/xls/test/"):
	image_data=cv2.imread("../../../data/xls/test/{}".format(file))
	preds=[]
	for i in range(6):
		for j in range(5):
			x_upper=min(448*(i+1),2560)
			y_upper=min(448*(j+1),1920)
			image_cut=image_data[y_upper-448:y_upper,x_upper-448:x_upper]
			image_cut=cv2.resize(image_cut,(224,224)).reshape(1,224,224,3)
			prediction=model.predict(image_cut)
			preds.append(prediction.ravel()[0])
	result=max(preds)
	files.append(file)
	predictions.append(result)

df_pred=pd.DataFrame({'filename':files,"probability":predictions})
df_pred.to_csv("./prediction.csv",index=False)


with open("./history.txt","w") as f:
    f.write(str(history.history))
