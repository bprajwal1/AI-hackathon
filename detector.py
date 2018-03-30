import pandas as pd
import numpy as np
import cv2
import os, sys
from tqdm import tqdm
from PIL import Image as ii

import xml.etree.ElementTree


import glob
path_xml = glob.glob("annots/*.xml")

test_images = []

def get_name(a):
	e = xml.etree.ElementTree.parse(a).getroot()
	result = [0, 0, 0, 0]
	image_path = ''
	flag = True
	count = 0
	for i in e:
		if i.tag == 'object':
			count += 1
	if count > 1:
		return [], 'x'

	for i in e:
		if i.tag == 'filename':
			image_path = i.text
		if i.tag == 'object':
			for j in i:
				if(j.tag == 'bndbox'):
					for k in j:
						if(k.tag == 'xmin'):
							result[1] = int(k.text) - 1
						elif(k.tag == 'ymin'):
							result[0] = int(k.text) - 1
						elif(k.tag == 'xmax'):
							result[3] = int(k.text) - 1
						else:
							result[2] = int(k.text) - 1
	print a, image_path, result
	return result, image_path



def read_img(img_path):
	img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
	img = np.array(img)
	return img

# take a sample image
img = read_img('jpeg/2007_000027.jpg')
IMG_SIZE = len(img)

for i in img:
	for j in i:
		j[0] = j[1] = j[2] = 0
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

data_set = []
count = 0
sample = 250
for i in path_xml:
	tmp = get_name(i)
	if(len(tmp[0]) == 0):
		continue
	img = read_img('jpeg/' + tmp[1])
	data_set.append((img, tmp[0]))
	test_images.append(img)
	sample -= 1
	if(sample == 0):
		break

print img
print len(img), len(img[0]), len(img[0][0])
train_X = [data_set[i][0] for i in range(len(data_set))]
train_X = np.array(train_X, np.float32) / 255
print 'shape ' ,train_X.shape
train_label = [data_set[i][1] for i in range(len(data_set))]
train_label = np.array(train_label)


import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

batch_size = 1
epochs = 10

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(IMG_SIZE,IMG_SIZE,3),padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2),padding='same'))
model.add(Conv2D(32, (3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))                
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Flatten())
model.add(Dense(32, activation='linear'))
model.add(LeakyReLU(alpha=0.1))                    
model.add(Dense(4, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])

train = model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1)

op = model.predict(train_X[:40])

import numpy as np

for i in range(10):
	print path_xml[i]
	print op[i]
	print train_label[i]
	print
	tmp = get_name(path_xml[i])
	img = test_images[i]
	x1, y1, x2, y2 = min(int(op[i][0]),499), min(int(op[i][1]), 499), min(int(op[i][2]),499), min(int(op[i][3]),499)
	y = min(y1, y2)
	while(y <= max(y1, y2) and y < IMG_SIZE):
		img[x1][y] = [0, 0, 0]
		img[x2][y] = [0, 0, 0]
		y += 1
	x = min(x1, x2)
	while(x <= max(x1, x2) and x < IMG_SIZE):
		img[x][y1] = [0, 0, 0]
		img[x][y2] = [0, 0, 0]
		x += 1
	cv2.imshow('image',test_images[i])
	cv2.waitKey(0)
	cv2.destroyAllWindows()












