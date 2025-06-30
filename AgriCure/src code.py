
import random
import numpy as np
import tensorflow as tf
import os    #handle file and dierectory operations
import json
from zipfile import ZipFile
import matplotlib.pyplot as plt  #for visualisation of images
import matplotlib.image as mpimg #to read image
from PIL import Image   # python imaging library-for openining and resizing images
from tensorflow.keras.preprocessing.image import ImageDataGenerator #for loading and augmenting images
from tensorflow.keras import layers,models  #to train and build model

random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)

# Setup Kaggle API to Col
kaggle=json.load(open("kaggle (1).json"))

os.environ['KAGGLE_USERNAME']=kaggle['username']
os.environ['KAGGLE_KEY']=kaggle['key']

import kaggle

!kaggle datasets download -d abdallahalidev/plantvillage-dataset

f1 = ZipFile("plantvillage-dataset.zip")   #extracts dataset to cwd
f1.extractall()
f1.close()

base_dir = "plantvillage dataset/color"

# Setting the base directory to a color folder
ipath="/content/plantvillage dataset/color/Cherry_(including_sour)___Powdery_mildew/012b55cf-de7a-483a-9118-0fd8ff012185___FREC_Pwd.M 0423.JPG"
img=mpimg.imread(ipath)
plt.imshow(img)
plt.axis("off")

img.shape

# Make all the images uniform
img_size=224
batch_size=32
data_gen=ImageDataGenerator(rescale=1./255,validation_split=0.2)  #normalising ixel values for faster training

# validation_split is testing size
train_generator=data_gen.flow_from_directory(base_dir,target_size=(img_size,img_size),batch_size=batch_size,class_mode="categorical",subset="training")

# validation_split is testing size
validation_generator=data_gen.flow_from_directory(base_dir,target_size=(img_size,img_size),batch_size=batch_size,class_mode="categorical",subset="validation")

# Making of neural network
model=models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(img_size,img_size,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(32,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dense(train_generator.num_classes,activation='softmax'))

model.summary()

np.array([[1,2,3]]).flatten()

# Pass the data to my network
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(train_generator,
          steps_per_epoch=train_generator.samples//batch_size,
          epochs=5,
          validation_data=validation_generator,
          validation_steps=validation_generator.samples//batch_size)

model.save("AgriCure.h5")
