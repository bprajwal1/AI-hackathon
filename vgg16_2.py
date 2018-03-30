from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
import os
import xml.etree.ElementTree as ET

arr = ["aeroplane", "bicycle", "bird", "boat", "bottle", "car", "cat", 
		"chair", "cow", "dog", "horse", "motorbike", "person",
		"train", "tvmonitor"]

path = "newann/"
impath = "VOCdevkit/VOC2010/JPEGImages/"
dirs = os.listdir(path)
im = []
la = []
i = 0

vec = [0 for i in range(15)]

for item in dirs:
	if i > 200:
		break
	if os.path.isfile(path+item): 
		#f, e = os.path.splitext(path+item)
		print(i, item)
		tree = ET.parse(path + item)
		root = tree.getroot()

		obj = root.find("object")
		vec[arr.index(obj[0].text)] = 1
		la.append(vec[:])
		vec[arr.index(obj[0].text)] = 0
		
		img = image.load_img(impath + item[:-4] + ".jpg", target_size=(224, 224))
		img_data = image.img_to_array(img)
		#img_data = np.expand_dims(img_data, axis=0)
		img_data = preprocess_input(img_data)
		print(img_data.shape)
		
		im.append(img_data)

		i += 1

im = np.asarray(im, dtype="float16")
print(im.shape)
la = np.asarray(la, dtype="int32")
print(la.shape)


base_model = VGG16(weights='imagenet')

x = base_model.output

#x = GlobalAveragePooling2D()(x)

x = Dense(1024, activation='relu')(x)

predictions = Dense(15, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(im, la, batch_size=8, nb_epoch=2)

for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

for layer in model.layers[:16]:
   layer.trainable = False
for layer in model.layers[16:]:
   layer.trainable = True

from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(im, la, batch_size=32, nb_epoch=1)

model.save("vgg_model3.h5")

img_path = 'testimage.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_data = image.img_to_array(img)
img_data = np.expand_dims(img_data, axis=0)
img_data = preprocess_input(img_data)

preds = model.predict(img_data)

print(preds)
