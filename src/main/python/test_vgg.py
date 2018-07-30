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
import keras.backend.tensorflow_backend as KTF

config = tf.ConfigProto()  
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
session = tf.Session(config=config)

# 设置session
KTF.set_session(session)
##设置模型参数
batch_size=16
steps_per_epoch=65
epochs=250

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)    


## 导入vgg模型，并固定所有卷积层
model_vgg=ResNet50(include_top=False,input_shape=(224,224,3),pooling=None)
for layer in model_vgg.layers[:-11]:
    layer.trainable=False

## 构造模型并进行编译
X=model_vgg.output #get_layer("activation_49").output
X=Flatten()(X)
X=Dropout(0.3)(X)
#X=Dense(128,activation="relu")(X)
Y=Dense(1,activation="sigmoid")(X)
model=Model(inputs=model_vgg.input,outputs=Y)

optimizer=Adam(lr=0.0005)
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["acc"])
model.summary()

## 设置data_augmentation
gen1=ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.1,
    channel_shift_range=0.1,
    fill_mode='nearest',
    horizontal_flip=True,
    vertical_flip=True,)
gen2=ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,)

train_gen=gen1.flow_from_directory("../../../data/xls/train",batch_size=batch_size,target_size=(224,224),
class_mode="binary",shuffle=True)

val_gen=gen2.flow_from_directory("../../../data/xls/val",target_size=(224, 224),
        batch_size=batch_size,class_mode='binary')
print("\nplease notice that the label and corresponding item:",train_gen.class_indices)
## 设置学习率递减

reduce_lr=ReduceLROnPlateau(monitor='val_acc',
factor=0.2,
patience=5,
verbose=1,
min_lr=1e-5)
def schedule(epoch):
	return 0.001*(1/(1+epoch))

lr_scheduler=LearningRateScheduler(schedule)

# 设置earlystop
early_stop=EarlyStopping(monitor='val_acc', patience=8, verbose=0, mode='auto')

#训练模型
history=model.fit_generator(train_gen,steps_per_epoch=steps_per_epoch,epochs=epochs,
			callbacks=[reduce_lr,early_stop],validation_data=val_gen,validation_steps=625)


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
			image_cut=cv2.resize(image_cut,(224,224))
			image_origin=image_cut.reshape(1,224,224,3)
			image_flip_h=cv2.flip(image_cut,1).reshape(1,224,224,3)
			image_flip_v=cv2.flip(image_cut,0).reshape(1,224,224,3)
			image_flip_hv=cv2.flip(image_cut,-1).reshape(1,224,224,3)
			
			prediction=(model.predict(image_origin)+model.predict(image_flip_h)+model.predict(image_flip_v)+model.predict(image_flip_hv))/4
			preds.append(prediction.ravel()[0])
	result=1-np.clip(np.mean(preds),0.001,0.999)
	files.append(file)
	predictions.append(result)

df_pred=pd.DataFrame({'filename':files,"probability":predictions})
df_pred.to_csv("./prediction.csv",index=False)


with open("./history.txt","w") as f:
    f.write(str(history.history))
