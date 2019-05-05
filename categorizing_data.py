import pandas as pd 
from glob import iglob
import os
import shutil
from tqdm import tqdm
import time

#LOAD CSV
tracks = pd.read_csv("fma_metadata/tracks.csv", header=[0,1], skiprows=[2]) 
# RENAME TRACK_ID COLUMN
tracks.rename(columns={'Unnamed: 0_level_0': 'track', 'Unnamed: 0_level_1': 'track_id'}, inplace=True) 

# WORKING ONLY WITH SMALL DATA
tracks = tracks[tracks['set','subset'] == 'small'] 
#LIST OF ALL UNIQUE GENRES
genre_top = pd.unique(tracks['track']['genre_top'])
genre_top = [i if i!=r'Old-Time / Historic' else 'Old' for i in genre_top]  
# CREATE NEW DIRECTORIES
if os.path.exists('data'):
	shutil.rmtree('data/')
	time.sleep(1)
	os.makedirs('data') 
# CREATE DIRECTORIES FOR EACH UNIQUE GENRE
for i in genre_top:
	os.makedirs('data/'+str(i)+'/')
# LIST OF ALL ABSOLUTE PATH OF ALL MP3s
abs_path = [f for f in iglob('**/*.mp3', recursive=True)]
# COPY FILES TO NEW GENRE FOLDERS
for file in tqdm(abs_path):
	name = int(os.path.splitext(os.path.basename(file))[0])
	genre = tracks['track']['genre_top'].loc[tracks['track']['track_id'] == name].values[0]
	if genre == r'Old-Time / Historic':
		genre = 'Old'
	shutil.copy(file, 'data/' + genre + '/')






