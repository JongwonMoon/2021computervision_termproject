import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from parameters import *
import numpy as np


train_datagen = ImageDataGenerator(
    rescale = 1./255,
    #zoom_range = 1.0,
    rotation_range = 20,
    height_shift_range = 0.1,
    width_shift_range = 0.1,
)

validation_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)

# 훈련 데이터 영상을 100x100 으로 resize
trainGen = train_datagen.flow_from_directory(
    os.path.join(rootPath, 'training_set'),
    target_size=(100, 100),
    class_mode='categorical',
    batch_size = BATCH_SIZE
)

# 검증 데이터 영상을 100x100 으로 resize
validationGen = validation_datagen.flow_from_directory(
    os.path.join(rootPath, 'validation_set'),
    target_size=(100, 100),
    class_mode='categorical',
    batch_size = BATCH_SIZE
)

testGen = test_datagen.flow_from_directory(
    os.path.join(rootPath, 'test_set'),
    target_size=(100, 100),
    class_mode='categorical',
    shuffle = False 
)
