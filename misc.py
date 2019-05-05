'''
import librosa
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences

arr1, sr = librosa.load('temp/001259.mp3')
arr2, sr2 = librosa.load('temp/006333.mp3')

#arr1 = arr1[0:50000]
#arr2 = arr2[0:50000]

#arr = np.append(arr, arr2)
arr = []
arr.append(arr1)
arr.append(arr2)

arr = np.asarray(arr)

print(arr.shape)

arr = pad_sequences(arr, dtype='float32', padding='post', truncating='post')

print(arr.shape)


from glob import glob
classes = {'Electronic':0, 'Experimental':1, 'Folk':2, 'Hip-Hop':3, 'Instrumental':4, 'International':5, 'Pop':6, 'Rock':7}

path = 'data/'
def load_file_paths_and_labels():
	labels = []
	for i in classes:
		for file in glob(path+i+'/*.mp3'):
			print(file)

load_file_paths_and_labels()
'''

import numpy as np
import pandas as pd 
from utils import DataGenerator


gen = DataGenerator(pd.read_csv('train_set.csv'), 2)

x,y = gen.__getitem__(1)

for i in x:
	print(i.shape)
print(y)
