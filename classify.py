import pandas as pd
import numpy as np
import cv2
import os, sys
from tqdm import tqdm

import xml.etree.ElementTree


import glob
path_xml = glob.glob("annots/*.xml")

def get_name(a):
	e = xml.etree.ElementTree.parse(a).getroot()
	result = []
	image_path = ''
	

	for i in e:
		if i.tag == 'filename':
			image_path = i.text
		if i.tag == 'object':
			for j in i:
				if(j.tag == 'name'):
					result.append(j.text)
	#print a, image_path, result
	return result, image_path


IMG_SIZE = 28
def read_img(img_path):
	img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
	img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
	img = np.array(img)
	return img

# take a sample image
img = read_img('jpeg/2007_000032.jpg')
data_set = []
class_id = dict()
id_class = []
count = 0
sample = 5000
for i in path_xml:
	tmp = get_name(i)
	for j in tmp[0]:
		if(j not in class_id):
			class_id[j] = count
			count += 1
			id_class.append(j)
	tt = [class_id[i] for i in tmp[0]]
	data_set.append((read_img('jpeg/' + tmp[1]), tt))
	sample -= 1
	if(sample == 0):
		break

print 'number of classes: ', len(id_class)
num_classes = len(id_class)

print img
print len(img), len(img[0]), len(img[0][0])
train_X = [data_set[i][0] for i in range(len(data_set))]
train_X = np.array(train_X, np.float32) / 255
print 'shape ' ,train_X.shape
train_label = []
for i in range(len(data_set)):
	train_label.append(list(range(num_classes)))
	for j in data_set[i][1]:
		train_label[i][j] = 1
train_label = np.array(train_label)


import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

batch_size = 64
epochs = 20

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(IMG_SIZE,IMG_SIZE,3),padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2),padding='same'))
model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))                  
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Flatten())
model.add(Dense(128, activation='linear'))
model.add(LeakyReLU(alpha=0.1))                  
model.add(Dense(num_classes, activation='sigmoid'))

model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

train = model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1)

y = model.predict(train_X[:10])

import numpy as np

for i in range(10):
	print path_xml[i]
	print id_class[np.argmax(y[i])], y[i]
	print












