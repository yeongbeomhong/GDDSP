
import glob
import os
import os.path
import time
import random
import numpy as np
import librosa
import scipy.io.wavfile
import jams
import pretty_midi
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

audio_path = "/data4/guitarset/audio_hex-pickup_debleeded/valid/"
sr = 22050
slice_length = 4
slice_hop = 0.2 

dir_of_each_audio_list = glob.glob(audio_path + "*")

for dir in tqdm(dir_of_each_audio_list):
    
    sec_duration = int(librosa.get_duration( filename = dir ))
    new_dir = dir[:-4] + f"_dur_{sec_duration}" + ".wav"
    os.rename(dir, new_dir)
    
    