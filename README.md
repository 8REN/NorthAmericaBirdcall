# birdcall_project
https://www.kaggle.com/c/birdsong-recognition - data source
if using mac, in order to process mp3s to wav, enter the follow into terminal:
% find . -name ".DS_Store" -print -delete

# preprocessing audio files

## audiomp3conversion.ipynb
files in mp3 format, resampled to wav format using pydub AudioSegment
old files archived for space conservation.
## audioStripSilence.ipynb
second step in audio processing to remove gaps with silence to condense audio 
to get as much usable audio as possible in less space, 
resampled all files to 32kHz. added all files to one folder rather than 
subfolders provided
###  to flush the librosa cache
librosa.cache.clear()
this is used in cases where storage reaches capacity
cache must be created prior to loading librosa: 
EXAMPLE:
import os
os.environ['LIBROSA_CACHE_DIR'] = '/tmp/librosa_cache'
import librosa
or see https://librosa.org/doc/latest/cache.html for more info
# preprocessDataVisualizations.ipynb
### libraries needed(also for audioVisualizations)
import os
os.environ['LIBROSA_CACHE_DIR'] = '/tmp/librosa_cache'
import librosa
from librosa import feature
import librosa.display 
from pathlib import Path
import audioread
import IPython
from scipy.io import wavfile
import pandas as pd 
import scipy
from scipy import stats
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import seaborn as sns
import shapefile as shp
from shapely.geometry import Point, Polygon
%matplotlib inline
import numpy as np
from tqdm import tqdm
## extracted season and time of day info
using date and time info provided
## opted to do north american recordings

## extracted length of recordings
eliminated recordings under 2 seconds in length
## images folder 
shapefiles for maps 
bird png images used in graphs


# audioVisualizations.ipynb
* create wavefile visualizations
* chromagrams
* chromagrams/cqt
* melspectograms
* extract mel and mfcc data and save for possible RNN/LTSM use
* onset plotting and evaluation

# featureExtraction
* get stft from file, to calculate the following:
* spectral contrast
* spectral deviation
* onset envelope
* energy (rms)
* mfccs
* melspectogram mean
* spectral rolloff


# modelData
* one hot encode target
* label encode target
* compare models
* kmeans clustering
* decision tree
* neural networks
   * bottleneck learning environment
   * flat environment
   * sparse categorial loss function for label encoded target
   * categorical crossentropy loss function for one hot encoded target
    

# to improve results:
* nlp on type and description from original csv
* divide countries by longitude and latitude into regions
* work on audio tracks to reduce background noise/level
* lstm using time series layer for complete spectogram/melspectogram extractions
* combine features extracted with cnn or lstm for better results?
* reassess audio used, relevel? split into smaller tracks?

