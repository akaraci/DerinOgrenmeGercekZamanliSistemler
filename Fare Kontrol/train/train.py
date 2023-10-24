import csv
import numpy as np 
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D,Flatten,Dense,Activation,Dropout,MaxPooling2D
from tensorflow.keras.optimizers import Adam

#26x34x1 resimleri kullanacağız
height = 26
width = 34
dims = 1

def readCsv(path):

	with open(path,'r') as f:
		# scv dosyasını sözlük biçimiyle okuma
		reader = csv.DictReader(f)
		rows = list(reader)

	# imgs, tüm görüntüleri içeren numpy dizisidir
	# tgs, resimlerin etiketlerini içeren numpy dizisidir
	imgs = np.empty((len(list(rows)),height,width, dims),dtype=np.uint8)
	tgs = np.empty((len(list(rows)),1))
		
	for row,i in zip(rows,range(len(rows))):

		# listeyi görüntü formatına geri dönüştürün
		img = row['image']
		img = img.strip('[').strip(']').split(', ')
		im = np.array(img,dtype=np.uint8)
		im = im.reshape((26,34))
		im = np.expand_dims(im, axis=2)
		imgs[i] = im

		# açma etiketi 1 ve kapatma etiketi 0
		tag = row['state']
		if tag == 'open':
			tgs[i] = 1
		else:
			tgs[i] = 0

	# veri kümesini karıştır
	index = np.random.permutation(imgs.shape[0])
	imgs = imgs[index]
	tgs = tgs[index]


	return imgs,tgs	


#evrişim sinir ağını yap
def makeModel():
	model = Sequential()

	model.add(Conv2D(32, (3,3), padding = 'same', input_shape=(height,width,dims)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Conv2D(64, (2,2), padding= 'same'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(128, (2,2), padding='same'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))

	
	model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy',metrics=['accuracy'])

	return model

def main():

	xTrain ,yTrain = readCsv('dataset.csv')
	print (xTrain.shape[0])
	# görüntülerin değerlerini 0 ile 1 arasında ölçeklendir
	xTrain = xTrain.astype('float32')
	xTrain /= 255

	model = makeModel()

	# biraz veri artırma yap
	datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.2,
        height_shift_range=0.2,
        )
	datagen.fit(xTrain)

	#modeli eğit
	model.fit_generator(datagen.flow(xTrain,yTrain,batch_size=32),
						steps_per_epoch=len(xTrain) / 32, epochs=50)
	
	#modeli kaydet
	model.save('blinkModel.hdf5')

if __name__ == '__main__':
	main()
