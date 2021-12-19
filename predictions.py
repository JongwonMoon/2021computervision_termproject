import os, scipy.io
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage import io, transform
from tensorflow import keras
import scipy.io
from parameters import *
from functions import testGen
import pandas as pd

#tf.enable_eager_execution()
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# model load
model = tf.keras.models.load_model(load_weight)
model.get_weights()

# model predict
pred = model.predict(testGen)

pred_list = []
for i in range(len(pred)):
    pred_list.append(np.argmax(pred[i]))

img_list = testGen.filenames
df = pd.DataFrame(list(zip(img_list, pred_list)), columns =['Name', 'pred'])
df.to_csv('C:/Users/Gahyun/OneDrive/바탕 화면/Moon/수업/submission.csv', index=False)
print("정확도 :", model.evaluate_generator(testGen)[1])
print(df)
