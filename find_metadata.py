import numpy as np
import pandas as pd
import librosa
import json 
from tqdm import tqdm
from threading import Thread

file_paths = pd.read_csv('train_set.csv')['path'].values

def write_to_json_mel(files):
	mel = []

	for fp in tqdm(files, position=0):
		x,_ = librosa.load(fp)
		x = librosa.util.fix_length(x, 661500)
		mel.append(librosa.feature.melspectrogram(y=x, sr=22050))

	mel = np.asarray(mel)

	mel_data = {'mel_max':np.max(mel),'mel_min':np.min(mel),'mel_mean':np.mean(mel), 'mel_std':np.std(mel)}
	with open('metadata/mel_data.json','w') as f:
		json.dump(mel_data, f)


def write_to_json_chroma(files):
	chroma = []

	for fp in tqdm(files, position=1):
		x,_ = librosa.load(fp)
		x = librosa.util.fix_length(x, 661500)
		chroma.append(librosa.feature.chroma_stft(y=x, sr=22050))

	chroma = np.asarray(chroma)

	chroma_data = {'chroma_max':np.max(chroma),'chroma_min':np.min(chroma),\
	'chroma_mean':np.mean(chroma), 'chroma_std':np.std(chroma)}

	with open('metadata/chroma_data.json','w') as f:
		json.dump(chroma_data, f)


def write_to_json_mfcc(files):
	mfcc = []

	for fp in tqdm(files, position=2):
		x,_ = librosa.load(fp)
		x = librosa.util.fix_length(x, 661500)
		mfcc.append(librosa.feature.mfcc(y=x, sr=22050))

	mfcc = np.asarray(mfcc)

	mfcc_data = {'mfcc_max':np.max(mfcc),'mfcc_min':np.min(mfcc),\
	'mfcc_mean':np.mean(mfcc), 'mfcc_std':np.std(mfcc)}

	with open('metadata/mfcc_data.json','w') as f:
		json.dump(mfcc_data, f)


def write_to_json_spec(files):
	spec = []

	for fp in tqdm(files, position=3):
		x,_ = librosa.load(fp)
		x = librosa.util.fix_length(x, 661500)
		spec.append(librosa.feature.spectral_contrast(y=x, sr=22050))

	spec = np.asarray(spec)

	spec_data = {'spec_max':np.max(spec),'spec_min':np.min(spec),\
	'spec_mean':np.mean(spec), 'spec_std':np.std(spec)}

	with open('metadata/spec_data.json','w') as f:
		json.dump(spec_data, f)


def write_to_json_tonn(files):
	tonn = []

	for fp in tqdm(files, position=4):
		x,_ = librosa.load(fp)
		x = librosa.util.fix_length(x, 661500)
		tonn.append(librosa.feature.tonnetz(y=x, sr=22050))

	tonn = np.asarray(tonn)

	tonn_data = {'tonn_max':np.max(tonn),'tonn_min':np.min(tonn),\
	'tonn_mean':np.mean(tonn), 'tonn_std':np.std(tonn)}

	with open('metadata/tonn_data.json','w') as f:
		json.dump(tonn_data, f)


t_mel    =  Thread(target=write_to_json_mel,    args=(file_paths,))
t_chroma =  Thread(target=write_to_json_chroma, args=(file_paths,))
t_mfcc   =  Thread(target=write_to_json_mfcc,   args=(file_paths,))
t_spec   =  Thread(target=write_to_json_spec,   args=(file_paths,))
t_tonn   =  Thread(target=write_to_json_tonn,   args=(file_paths,))

t_mel.start()
t_chroma.start()
t_mfcc.start()
t_spec.start()
t_tonn.start()


t_mel.join()
t_chroma.join()
t_mfcc.join()
t_spec.join()
t_tonn.join()






