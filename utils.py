import librosa
import numpy as np
import pandas as pd
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import ZeroPadding2D, Conv2D, BatchNormalization, ReLU, DepthwiseConv2D
import json

#classes = {'Electronic':0, 'Experimental':1, 'Folk':2, 'Hip-Hop':3, 'Instrumental':4, 'International':5, 'Pop':6, 'Rock':7}

metadata = json.loads(open('metadata.json').read())

class DataGenerator(keras.utils.Sequence):

	def __init__(self, csv_df, batch_size, n_classes=8, shuffle=True):

		self.batch_size = batch_size 
		self.csv_df = csv_df 
		self.n_classes = n_classes 
		self.shuffle = shuffle 
		self.on_epoch_end()

	def __len__(self):
		return int(len(self.csv_df) / self.batch_size)
		

	def on_epoch_end(self):

		self.indexes = np.arange(len(self.csv_df))

		if self.shuffle == True:
			np.random.shuffle(self.indexes)

	def __getitem__(self, index):

		indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
		file_names = [self.csv_df['path'][k] for k in indexes]
		x,y = self.__data_generation(file_names)

		return x,y

	def __data_generation(self, file_names):

		batch_input = []
		batch_output = []
		for input_path in file_names:
			
			inp, _ = librosa.load(input_path, res_type='kaiser_fast')
			out = self.csv_df['labels'].loc[self.csv_df['path'] == input_path].values[0]

			batch_input.append(inp)
			batch_output.append(out)

		#PADDING ALL LOADED AUDIO FILES TO EXACTLY 30 SECONDS
		batch_x = pad_sequences(np.asarray(batch_input), maxlen=661500, dtype='float32', padding='pre', truncating='post')

		batch_x_mel = []
		batch_x_chroma = []
		batch_x_mfcc = []
		batch_x_spec = []
		batch_x_tonnetz = []

		#EXTRACTING FEATURES 
		for i in batch_x:
			batch_x_mel.append(np.expand_dims(librosa.power_to_db(librosa.feature.melspectrogram(y=i, sr=22050)), axis=-1))
			batch_x_chroma.append(np.expand_dims(librosa.feature.chroma_stft(y=i, sr=22050), axis=-1))
			batch_x_mfcc.append(np.expand_dims(librosa.feature.mfcc(y=i, sr=22050), axis=-1))
			batch_x_spec.append(np.expand_dims(librosa.feature.spectral_contrast(y=i, sr=22050), axis=-1))
			batch_x_tonnetz.append(np.expand_dims(librosa.feature.tonnetz(y=i, sr=22050), axis=-1))
		
		batch_x_mel = np.asarray(batch_x_mel)
		batch_x_chroma = np.asarray(batch_x_chroma)
		batch_x_mfcc = np.asarray(batch_x_mfcc)
		batch_x_spec = np.asarray(batch_x_spec)
		batch_x_tonnetz = np.asarray(batch_x_tonnetz)

		#NORMALIZE

		batch_x_mel      = (batch_x_mel - metadata['mel']['mel_min'])/(metadata['mel']['mel_max'] - metadata['mel']['mel_min'])
		batch_x_mfcc     = (batch_x_mfcc - metadata['mfcc']['mfcc_min'])/(metadata['mfcc']['mfcc_max'] - metadata['mfcc']['mfcc_min'])
		batch_x_spec     = (batch_x_spec - metadata['spec']['spec_min'])/(metadata['spec']['spec_max'] - metadata['spec']['spec_min'])
		batch_x_tonnetz  = (batch_x_tonnetz - metadata['tonn']['tonn_min'])/(metadata['tonn']['tonn_max'] - metadata['tonn']['tonn_min'])


		#CONVERTING LABELS TO ONE-HOT
		batch_y = to_categorical(batch_output, self.n_classes)

		return [batch_x_mel, batch_x_chroma, batch_x_mfcc, batch_x_spec, batch_x_tonnetz], batch_y
		#return [batch_x_mel, batch_x_chroma, batch_x_mfcc], batch_y

