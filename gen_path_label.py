import pandas as pd 
from glob import iglob
import os
from tqdm import tqdm
from sklearn.utils import shuffle
import librosa

# DEFINE CLASSES
classes = {'Hip-Hop':0, 'Pop':1, 'Rock':2, 'Experimental':3, 'Folk':4, 'Jazz':5, 'Electronic':6, 'Spoken':7, \
		'International':8, 'Soul-RnB':9, 'Blues':10, 'Country':11, 'Classical':12, 'Old-Time / Historic':13, 'Instrumental':14, 'Easy Listening':15}

#LOAD CSV
tracks = pd.read_csv("fma_metadata/tracks.csv", header=[0,1], skiprows=[2]) 
# RENAME TRACK_ID COLUMN
tracks.rename(columns={'Unnamed: 0_level_0': 'track', 'Unnamed: 0_level_1': 'track_id'}, inplace=True) 
# WORKING ONLY WITH LABELLED DATA
tracks = tracks[tracks['track','genre_top'].notnull()] 

# LIST OF ALL ABSOLUTE PATH OF ALL MP3s
abs_path = [f for f in iglob('fma_large/**/*.mp3', recursive=True)]
labels = []
file_path = []
del_files = []

for file in tqdm(abs_path):
	try:
		librosa.load(file)
		name = int(os.path.splitext(os.path.basename(file))[0])
		genre = tracks['track']['genre_top'].loc[tracks['track']['track_id'] == name].values[0]
		labels.append(classes[genre])
		file_path.append(file)
	except:
		del_files.append(file)
		os.remove(file)


df = {'path': file_path, 'labels': labels}
df = pd.DataFrame(data=df)

df = shuffle(df)

train_df = df.sample(frac=0.98, random_state=1806)
test_df = df.drop(train_df.index)

train_df.to_csv('train_set.csv', index=False)
test_df.to_csv('valid_set.csv', index=False )

print('SUMMARY')
print('TRAIN DATA = ', str(len(train_df)))
print('VALIDATION DATA = ', str(len(test_df)))
print('DELETED FILES = ', str(len(del_files)))

