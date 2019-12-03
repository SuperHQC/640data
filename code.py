#!/usr/bin/env python
# coding: utf-8

# In[3]:


import cv2
import csv
import numpy as np
import keras
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.models import load_model
from keras import backend as K

import random


# In[4]:


import tensorflow as tf
K.tensorflow_backend._get_available_gpus()


# In[5]:


final_label = np.loadtxt('final_label.csv',delimiter = ',')
print(final_label)


# In[6]:



image_size = 64
def resize_img(image, height = image_size, width = image_size):
    top,bottom, left, right = (0,0,0,0)
    
    h,w,_=image.shape
    
    longest_edge = max(h,w)
    
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass
    
    BLACK = [0,0,0]
    constant = cv2.copyMakeBorder(image, top,bottom, left,right, cv2.BORDER_CONSTANT,value = BLACK)
    
    return cv2.resize(constant, (height,width))


# In[7]:


images = []


for i in range(len(final_label)):
# for i in range(1,2):
    img_name = str(i) +'.jpg'
    image = cv2.imread('./data/'+img_name)
#     cv2.imshow('image', np.array(image,dtype=np.uint8))
#     print(image[45,45,1])
    if image is None:
        del final_label[i]
    else:
        image = resize_img(image, image_size, image_size)
#         cv2.imshow('test', np.array(image,dtype=np.uint8))
#         print(image[30,30,1])
        images.append(image)
#     break
        
print(len(images))
print(len(final_label))


# In[8]:


images = np.array(images, dtype='float')
final_label = np.array(final_label)

print(images.shape)
print(final_label.shape)


# In[9]:


train_images, test_images, train_labels, test_labels = train_test_split(images, final_label, test_size = 0.3, random_state = random.randint(0,100))
print(test_labels)


# In[10]:


img_channels = 3
if K.image_data_format == 'channel_first':
    train_images = train_images.reshape(train_images.shape[0],img_channels, image_size, image_size)
    test_images = test_images.reshape(test_images.shape[0],img_channels, image_size, image_size)
    input_shape = (img_channels, image_size, image_size)
else:
    train_images = train_images.reshape(train_images.shape[0], image_size, image_size, img_channels)
    test_images = test_images.reshape(test_images.shape[0], image_size, image_size, img_channels)
    input_shape = (image_size, image_size, img_channels)

train_images /= 255
test_images /= 255


# In[11]:


class Model:
    def __init__(self):
        self.model = None
    def build_model(self, input_shape, nb_classes=3):
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), padding = 'same', input_shape = input_shape))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(32, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size = (2,2)))
        self.model.add(Conv2D(64, (3, 3), padding = 'same'))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size = (2,2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(nb_classes))
        self.model.add(Activation('softmax'))
        
    def train(self, train_images, train_labels, batch_size = 128, nb_epoch = 15, data_augmentation = True):
        self.model.compile(loss = 'categorical_crossentropy', 
                           optimizer = 'ADAM',
                           metrics = ['accuracy'])
        if not data_augmentation:
            self.model.fit(train_images, 
                           train_labels, 
                           batch_size = batch_size,
                           epochs = nb_epoch, 
                           shuffle = True)
        else:
            datagen = ImageDataGenerator(rotation_range = 20,
                                        width_shift_range = 0.2,
                                        height_shift_range = 0.2,
                                        horizontal_flip = True)
            
            self.model.fit_generator(datagen.flow(train_images, train_labels, batch_size = batch_size), epochs = nb_epoch)
            
    def evaluate(sefl, test_images, test_labels):
        score = self.model.evaluate(test_images, test_labels)
        print("%s:%.3f%%" % (self.model.metrics_names[1], score[1]*100))
        


# In[12]:


model = Model()
model.build_model(input_shape,3)


# In[13]:


model.train(train_images, train_labels, 128,20,True)


# In[ ]:


model.evaluate(test_images, test_labels)

