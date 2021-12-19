import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.python.keras import layers, applications, models, preprocessing, callbacks, optimizers
import tensorflow as tf
from model import MobileNetV2
from parameters import *
from functions import trainGen, validationGen
#import matplotlib.pyplot as plt
#from skimage import io, transform
from tensorflow.keras.applications import MobileNetV2, ResNet50
#from tensorflow.keras.models import Sequential

#Select GPUs
os.environ["CUDA_VISIBLE_DEVICES"]= '0'

#model = MobileNetV2(input_shape=(200, 200, 3), k=num_classes)
model = tf.keras.models.Sequential()
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
base_model.summary()
base_model.trainable = False  
inputs = tf.keras.Input(shape = IMG_SHAPE)
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(3, activation = 'softmax')(x)
model = tf.keras.Model(inputs, outputs)

model.summary()

ckpt_dir = os.path.dirname(ckpt_path)
checkpoint = callbacks.ModelCheckpoint(filepath = ckpt_path, monitor = 'val_loss', verbose =1,
                                            save_best_only = True, mode='auto')
#early_stopping = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
#binary_crossentropy
model.compile(loss ='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

# training
#model.fit(trainGen, epochs=epochs, validation_data = validationGen, 
#          callbacks = [checkpoint,early_stopping])
model.fit_generator(trainGen,
                    epochs = 50,
                    verbose = 1,
                    validation_data = validationGen,
                    callbacks = [checkpoint])







