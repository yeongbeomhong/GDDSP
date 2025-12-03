# This file offers dataset class which loads audio files and ground-truth annotations from GuitarSet()
# the returned_tuple of __getitem__ is "(audiotensor, data_idx, pitch annotation)" 

import glob
import os
import os.path
import json
import time
import random
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
#from pedalboard import Distortion as spotify_distortion


class Dataset_Dry_Slice_Overlap_Realtime(Dataset):
    def __init__(self, config, device, audio_path = "/data4/guitarset/audio_hex-pickup_debleeded/train/"):
        self.audio_path = audio_path
        self.config = config
        self.slice_hop = config.slice_hop
        self.slice_length = config.slice_length
        self.sr = config.sr
        self.input_dist_amount_random = config.input_dist_amount_random
        self.input_dist_type = config.input_dist_type
        self.device = device
        
        self.preprocessed_dataset = self.long_audio_wav_to_sliced_tensor()
    def dir_of_each_audio_list(self) :
        return glob.glob(self.audio_path + "*.wav")
    
    def dataset_len_before_slicing(self):
        return len(self.dir_of_each_audio_list())
        
    def num_of_sliced_excerpt_of_each_source(self):
        # Use this function only if there is duration information in filename. 
        num_of_excerpt_list = []
        for dir in self.dir_of_each_audio_list() :
            filename = dir[:-4] # eliminate the string ".wav"
            source_duration = int(filename.split('_')[-1])
            num_slice = int((source_duration-self.slice_length)/self.slice_hop) + 1
            num_of_excerpt_list.append(num_slice)
        return num_of_excerpt_list
            
    def cumulative_num_of_sliced_excerpt(self):
        information_already_exist_path = self.audio_path + 'cumulative_num.json'
        if not os.path.exists(information_already_exist_path):
            cumulative_num_of_excerpt_list = []
            cumulative_num = 0
            for dir in self.dir_of_each_audio_list() :
                filename = dir[:-4] # eliminate the string ".wav"
                source_duration = int(filename.split('_')[-1])
                num_slice = int((source_duration-self.slice_length)/self.slice_hop) + 1
                cumulative_num += num_slice
                cumulative_num_of_excerpt_list.append(cumulative_num)
            with open(information_already_exist_path, 'w') as json_file:
                json.dump(cumulative_num_of_excerpt_list, json_file)
        else: # There already exists cumulative num of overlapping sliced excerpts for whole audio dataset
            with open(information_already_exist_path , 'r') as json_file:
                cumulative_num_of_excerpt_list = json.load(json_file)
        return cumulative_num_of_excerpt_list
    def long_audio_wav_to_sliced_tensor(self):
        import time
        start_slicing =time.time()
        sliced_tensors_filename = self.audio_path + f"length{self.slice_length}_hop{self.slice_hop}_all_slice_tensors.pt"
        if os.path.exists(sliced_tensors_filename):
            slice_tensors_of_all_sources = torch.load(sliced_tensors_filename)
            if slice_tensors_of_all_sources.device != "cpu":
                slice_tensors_of_all_sources = slice_tensors_of_all_sources.cpu()
            print(f"Time Consumed to load {slice_tensors_of_all_sources.shape[0]} sliced tensors previously made from \
                {self.dataset_len_before_slicing()} long audios : {time.time()-start_slicing}")
        else:
            dir_of_each_audio_list = self.dir_of_each_audio_list()
            for source_idx, source_dir in enumerate(tqdm(dir_of_each_audio_list, desc="Slice_and_Tensorize_audio")):
                source_wav, sr = librosa.load(source_dir, sr = self.sr, mono = True)
                source_tensor = torch.tensor(source_wav, dtype=torch.float32).to(self.device)
                slice_tensors_of_ith_source = source_tensor.unfold(-1, self.sr*self.slice_length, self.sr*self.slice_hop)
                if not source_idx == 0:
                    slice_tensors_of_all_sources = torch.cat((slice_tensors_of_all_sources, slice_tensors_of_ith_source), dim=-2)
                    #dim = -1 -> sample points axis. 22050*4 points per one slice
                    #dim = -2 -> slice_idx axis
                else : #when slicing first source audio
                    slice_tensors_of_all_sources = slice_tensors_of_ith_source
            torch.save(slice_tensors_of_all_sources.cpu() , sliced_tensors_filename)
            print(f"Time Consumed to process {self.dataset_len_before_slicing()} long audios into \
                {slice_tensors_of_all_sources.shape[0]}sliced tensors : {time.time()-start_slicing}")
        return slice_tensors_of_all_sources
    
    def __len__(self):
        #return 256 # small data num for short heuristic experiment
        return self.cumulative_num_of_sliced_excerpt()[-1] # The total number of excerpts is same as cumulative sum of exceprts at last long audio.
    
    def __getitem__(self, idx): # load the audio ignoring hexaphonic-property (regard the audio as the superposition of all pickup-cahnnels)
        cumulative_num_slice_list = self.cumulative_num_of_sliced_excerpt()

        # Find out how many source_audios are in front of the target source audio containing the sliced_audio corresponding to idx
        for source_idx, value in enumerate(cumulative_num_slice_list):
            if value > idx:
                break
        if source_idx == 0 :
            slice_idx = idx
        else :
            slice_idx = idx - cumulative_num_slice_list[source_idx-1]
            # Convert the Overall slice_idx amomg all slices, into local slice_idx in target source_audio.

        source, sr = librosa.load(self.dir_of_each_audio_list()[source_idx], sr = self.sr, mono = True)
        source_tensorized = torch.tensor(source, dtype=torch.float32)
        slice = source_tensorized[int(slice_idx*sr*self.slice_hop): int(slice_idx*sr*self.slice_hop + self.slice_length*sr)]
        return {"audio_dry_GT": slice}
    

class Dataset_Dist_Slice_Overlap_Realtime(Dataset):
    def __init__(self, config, device, audio_path = "/data4/guitarset/audio_hex-pickup_debleeded/train/"):
        self.config = config
        self.slice_hop = config.slice_hop
        self.slice_length = config.slice_length
        self.sr = config.sample_rate
        self.input_dist_amount_random = config.input_dist_amount_random
        self.input_dist_type = config.input_dist_type
        self.input_dist_amount_fixed = config.input_dist_amount_fixed
        self.device = device
        self.audio_path = audio_path

        if self.input_dist_type == "torch_overdrive" or self.input_dist_type == "spotify_pedalboard":
            self.preprocessed_dataset = self.long_audio_wav_to_sliced_tensor()
        else:
            raise ValueError("You must pick valid input_dist_type in config file.")
        
    def dir_of_each_audio_list(self) :
        return glob.glob(self.audio_path + "*.wav")

    def dataset_len_before_slicing(self):
        return len(self.dir_of_each_audio_list())
    
    def num_of_sliced_excerpt_of_each_source(self):
        # Use this function only if there is duration information in filename. 
        num_of_excerpt_list = []
        for dir in self.dir_of_each_audio_list() :
            filename = dir[:-4] # eliminate the string ".wav"
            source_duration = int(filename.split('_')[-1])
            num_slice = int((source_duration-self.slice_length)/self.slice_hop) + 1
            num_of_excerpt_list.append(num_slice)
        return num_of_excerpt_list
    def cumulative_num_of_sliced_excerpt(self):
        information_already_exist_path = self.audio_path + 'cumulative_num.json'
        if not os.path.exists(information_already_exist_path):
            cumulative_num_of_excerpt_list = []
            cumulative_num = 0
            for dir in self.dir_of_each_audio_list() :
                filename = dir[:-4] # eliminate the string ".wav"
                source_duration = int(filename.split('_')[-1])
                num_slice = int((source_duration-self.slice_length)/self.slice_hop) + 1
                cumulative_num += num_slice
                cumulative_num_of_excerpt_list.append(cumulative_num)
            with open(information_already_exist_path, 'w') as json_file:
                json.dump(cumulative_num_of_excerpt_list, json_file)
        else: # There already exists cumulative num of overlapping sliced excerpts for whole audio dataset
            with open(information_already_exist_path , 'r') as json_file:
                cumulative_num_of_excerpt_list = json.load(json_file)
        return cumulative_num_of_excerpt_list
    
    def __len__(self):
        #return 32 #small data num for short heuristic experiment
        return self.preprocessed_dataset.shape[0]

    def long_audio_wav_to_sliced_tensor(self):
        import time
        start_slicing =time.time()
        sliced_tensors_filename = self.audio_path + f"length{self.slice_length}_hop{self.slice_hop}_all_slice_tensors.pt"
        if os.path.exists(sliced_tensors_filename):
            slice_tensors_of_all_sources = torch.load(sliced_tensors_filename)
            if slice_tensors_of_all_sources.device != "cpu":
                slice_tensors_of_all_sources = slice_tensors_of_all_sources.cpu()
            print(f"Time Consumed to load {slice_tensors_of_all_sources.shape[0]} sliced tensors previously made from \
                {self.dataset_len_before_slicing()} long audios : {time.time()-start_slicing}")
        else:
            dir_of_each_audio_list = self.dir_of_each_audio_list()
            for source_idx, source_dir in enumerate(tqdm(dir_of_each_audio_list, desc="Slice_and_Tensorize_audio")):
                source_wav, sr = librosa.load(source_dir, sr = self.sr, mono = True)
                source_tensor = torch.tensor(source_wav, dtype=torch.float32).to(self.device)
                slice_tensors_of_ith_source = source_tensor.unfold(-1, self.sr*self.slice_length, self.sr*self.slice_hop)
                if not source_idx == 0:
                    slice_tensors_of_all_sources = torch.cat((slice_tensors_of_all_sources, slice_tensors_of_ith_source), dim=-2)
                    #dim = -1 -> sample points axis. 22050*4 points per one slice
                    #dim = -2 -> slice_idx axis
                else : #when slicing first source audio
                    slice_tensors_of_all_sources = slice_tensors_of_ith_source
            torch.save(slice_tensors_of_all_sources.cpu() , sliced_tensors_filename)
            print(f"Time Consumed to process {self.dataset_len_before_slicing()} long audios into \
                {slice_tensors_of_all_sources.shape[0]}sliced tensors : {time.time()-start_slicing}")
        return slice_tensors_of_all_sources

    def __getitem__(self, idx): # load the audio ignoring hexaphonic-property (regard the audio as the superposition of all pickup-cahnnels)
        slice = self.preprocessed_dataset[idx]
        
        if self.input_dist_type == "torch_overdrive" : 
            if self.input_dist_amount_random == True :
                gain = random.uniform(5, 30)
                colour = 20
                slice_distorted = torchaudio.functional.overdrive(slice, gain=gain, colour=colour) * 0.9
                #Fit Model to infer Random Various Distortion # TO DO Mix ratio
            else:
                gain = self.input_dist_amount_fixed
                colour = 20
                slice_distorted = torchaudio.functional.overdrive(slice, gain=gain, colour=colour) * 0.9 #Fit Model to one fixed Distortion
            return {"audio_dry_GT": slice, "audio_wet_GT":slice_distorted, "gain_GT" : gain}
    
        elif self.input_dist_type == "spotify_pedalboard" :
            if self.input_dist_amount_random == True :
                gain = random.uniform(5, 30)
                processor = spotify_distortion(drive_db = gain)
                slice_distorted = processor(slice.numpy(), self.sr)
                slice_distorted = torch.tensor(slice_distorted) * 0.9
                #Fit Model to infer Random Various Distortion # TO DO Mix ratio
            else:
                gain = self.input_dist_amount_fixed
                processor = spotify_distortion(drive_db = gain)
                slice_distorted = processor(slice.numpy(), self.sr)
                slice_distorted = torch.tensor(slice_distorted) * 0.9 #Fit Model to one fixed Distortion
            return {"audio_dry_GT": slice, "audio_wet_GT":slice_distorted, "gain_GT" : gain}
        else:
            raise ValueError("config.input_dist_type must be torch_overdrive or spotifiy_pedalboard or mix.")
        '''
        elif self.input_dist_type == "mix" :
            if random.choice([0,1]) == 0 : # torch_overdrive is chosen !
                if self.input_dist_amount_random == True :
                    gain = random.uniform(5, 30)
                    colour = random.uniform(5, 30)
                    slice_distorted = torchaudio.functional.overdrive(slice, gain=gain, colour=colour) * 0.9
                    #Fit Model to infer Random Various Distortion # TO DO Mix ratio
                else:
                    gain = self.input_dist_amount_fixed
                    colour = 20
                    slice_distorted = torchaudio.functional.overdrive(slice, gain=gain, colour=colour) * 0.9 #Fit Model to one fixed Distortion
                return {"audio_dry_GT": slice, "audio_wet_GT":slice_distorted, "gain_GT" : gain}
            else : # spotify_pedalboard is chosen !
                if self.input_dist_amount_random == True :
                    gain = random.uniform(5, 30)
                    processor = spotify_distortion(drive_db = gain)
                    slice_distorted = processor(slice.numpy(), self.sr)
                    slice_distorted = torch.tensor(slice_distorted) * 0.9
                    #Fit Model to infer Random Various Distortion # TO DO Mix ratio
                else:
                    gain = self.input_dist_amount_fixed
                    processor = spotify_distortion(drive_db = gain)
                    slice_distorted = processor(slice.numpy(), self.sr)
                    slice_distorted = torch.tensor(slice_distorted) * 0.9 #Fit Model to one fixed Distortion
                return {"audio_dry_GT": slice, "audio_wet_GT":slice_distorted, "gain_GT" : gain}
        '''
        
            
class Dataset_Dist_Slice_Overlap_Realtime_Stringwise(Dataset):
    # Dataset to train the Monophonic DDSP
    # Load the GuitarSet Trainset Audio Conserving 6-channel structures. (librosa mono argument: false)
    # Then separate string-by-string. Get 6 long audio per source audio. (train: 300audio -> 1800audio,  valid:60->360)
    # slice each audio, tensorize, Make [4130*6, 88200] train tensor.pt and [826*6, 88200] valid tensor.pt  
    # (24780 train string-wise slices,  4956 valid string-wise slices)
    # Sort slices with ordering that locate "slices of six strings of same timesteps of one audio" to be adjacent
    # (In getitem, loading order is source1_slice1_low E string, source1_slice1_low A string, ... , Rather than order of source_slice1_low E string -> source1_slice2_low E string)
    def __init__(self, config, device, audio_path = "/data4/guitarset/audio_hex-pickup_debleeded/train/"):
        self.config = config
        self.slice_hop = config.slice_hop
        self.slice_length = config.slice_length
        self.sr = config.sample_rate
        self.input_dist_amount_random = config.input_dist_amount_random
        self.input_dist_type = config.input_dist_type
        self.input_dist_amount_fixed = config.input_dist_amount_fixed
        self.device = device
        self.audio_path = audio_path

        if self.input_dist_type == "torch_overdrive" or self.input_dist_type == "spotify_pedalboard":
            self.preprocessed_dataset = self.long_audio_wav_to_sliced_tensor()
        else:
            raise ValueError("You must pick proper input_dist_type in config file.")
        
    def dir_of_each_audio_list(self) :
        return glob.glob(self.audio_path + "*.wav")

    def dataset_len_before_slicing(self):
        return len(self.dir_of_each_audio_list())
    
    def __len__(self):
        #return 72 #small data num for short heuristic experiment
        return self.preprocessed_dataset.shape[0]

    def long_audio_wav_to_sliced_tensor(self):
        import time
        start_slicing =time.time()
        sliced_tensors_filename = self.audio_path + f"stringwise_length{self.slice_length}_hop{self.slice_hop}_all_slice_tensors.pt"
        if os.path.exists(sliced_tensors_filename):
            slice_tensors_of_all_sources = torch.load(sliced_tensors_filename)
            if slice_tensors_of_all_sources.device != "cpu":
                slice_tensors_of_all_sources = slice_tensors_of_all_sources.cpu()
            print(f"Time Consumed to load {slice_tensors_of_all_sources.shape[0]} sliced string-wise tensors previously made from \
                {self.dataset_len_before_slicing()} long audios : {time.time()-start_slicing}")
        else:
            dir_of_each_audio_list = self.dir_of_each_audio_list()
            for source_idx, source_dir in enumerate(tqdm(dir_of_each_audio_list, desc="Slice_and_Tensorize_audio")):
                source_wav_all_strings, sr = librosa.load(source_dir, sr = self.sr, mono = False) #mono option must be False
                if source_wav_all_strings.shape[0] != 6 :
                    raise ValueError("Loading Guitarset.wav with librosa mono=Falase when source has hexa-channel -> must return numpy array with shape (6, ...)")
                else:
                    pass
                source_tensor_all_strings = torch.tensor(source_wav_all_strings, dtype=torch.float32).to(self.device)
                slice_tensors_ith_source_all_strings = source_tensor_all_strings.unfold(-1, self.sr*self.slice_length, self.sr*self.slice_hop) 
                # Output shape of unfold :(6, slice_nums, 4sec*sr)
                # slice_nums from each sourch audio can differ, so slices from audios cannot be stacked in this form.
                # Flatten the dim of (6,slice_nums) -> (6*slice_nums), 
                # Do not collect slices from 0th string-> slices from 1th string ...,  Instead, collect strings from 0th slice, strings from 1th slice...)
                # (Later you can make polyphonic superposition of recon slices from 6strings of same audio by sum batch audio with period of six
                # It menas, Flatten string-axis and slice-per-string-axis with column-centric Ordering which is reverse of Default Ordering(row-ordering)
                slice_num = slice_tensors_ith_source_all_strings.shape[1]
                slice_tensors_ith_source_strings_of_same_timestep_adjacent = slice_tensors_ith_source_all_strings.transpose(0,1).contiguous().view(6*slice_num , self.sr*self.slice_length)
                if source_idx == 0 :
                    slice_tensors_of_all_sources = slice_tensors_ith_source_strings_of_same_timestep_adjacent 
                else:
                    slice_tensors_of_all_sources = torch.cat((slice_tensors_of_all_sources, slice_tensors_ith_source_strings_of_same_timestep_adjacent ), dim=0) # cat with slice_idx dim (not sample points dim)
            torch.save(slice_tensors_of_all_sources.cpu() , sliced_tensors_filename)
            print(f"Time Consumed to Tensorize {self.dataset_len_before_slicing()} hexaphonic long audios into \
                {slice_tensors_of_all_sources.shape[0]} sliced string-wise tensors : {time.time()-start_slicing}")
        return slice_tensors_of_all_sources.cpu()

    def __getitem__(self, idx): # load the audio ignoring hexaphonic-property (regard the audio as the superposition of all pickup-cahnnels)
        slice = self.preprocessed_dataset[idx] #sliced tensor of ith one-string audio.
        if self.input_dist_type == "torch_overdrive" : 
            if self.input_dist_amount_random == True :
                gain = random.uniform(5, 30)
                colour = 20
                slice_distorted = torchaudio.functional.overdrive(slice, gain=gain, colour=colour) * 0.9
                #Fit Model to infer Random Various Distortion # TO DO Mix ratio
            else:
                gain = self.input_dist_amount_fixed
                colour = 20
                slice_distorted = torchaudio.functional.overdrive(slice, gain=gain, colour=colour) * 0.9 #Fit Model to one fixed Distortion
            return {"audio_dry_GT": slice, "audio_wet_GT":slice_distorted, "gain_GT" : gain}
        elif self.input_dist_type == "spotify_pedalboard" :
            if self.input_dist_amount_random == True :
                gain = random.uniform(5, 30)
                processor = spotify_distortion(drive_db = gain)
                slice_distorted = processor(slice.numpy(), self.sr)
                slice_distorted = torch.tensor(slice_distorted) * 0.9
                #Fit Model to infer Random Various Distortion # TO DO Mix ratio
            else:
                gain = self.input_dist_amount_fixed
                processor = spotify_distortion(drive_db = gain)
                slice_distorted = processor(slice.numpy(), self.sr)
                slice_distorted = torch.tensor(slice_distorted) * 0.9 #Fit Model to one fixed Distortion
            return {"audio_dry_GT": slice, "audio_wet_GT":slice_distorted, "gain_GT" : gain}
        else:
            raise ValueError("config.input_dist_type must be torch_overdrive or spotifiy_pedalboard or mix.")
        
class Test_Dataset_IDMT_Dist(Dataset):
    def __init__(self, config, device, dist = "808", mode = None): 
        #dist = "od1" , "808", "mgs", "rat"
        # mode : "test"/ "train"/ "valid"
        # mode-test: use whole 10000 dist audio in 808(or mgs or od1 or rat) as test dataset, for model pre-trained with guitarset and torch.overdrive
        # mode-train : use 3/4 (7500) dist audio to train new scratch model
        # mode-valid : use 1/4 (2500) dist audio to evaluate model undergoing training epochs, periodically 
        self.config = config
        self.sr = config.sample_rate
        self.device = device
        self.dist = dist
        self.path_wet = "/data4/idmt_dist_guitar/" + self.dist + "/"
        self.path_dry = "/data4/idmt_dist_guitar/NoFX/"
        if not os.path.exists(self.path_wet):
            raise ValueError("Select proper dist argument for idmt")
        if not os.path.exists(self.path_dry):
            raise ValueError("There is no NoFX Folder")
        if mode == None or mode == "test" :
            self.preprocessed_dataset_wet = self.wav_to_tensor_without_name_and_collect(path = self.path_wet)[7501:10000]
            self.preprocessed_dataset_dry = self.wav_to_tensor_without_name_and_collect(path = self.path_dry)
            self.preprocessed_name_list_wet , self.preprocessed_name_list_dry = self.get_sorted_filename_essence_list()
            self.preprocessed_name_list_wet = self.preprocessed_name_list_wet[7501:10000]
            # gutiarset에서 훈련/밸리드한 뒤 테스트만 IDMT에서 하는 경우
            # 이 때 test set = IDMT에서 훈련/밸리드 할 경우의 valid set과 
            # 양 실험 간에 공평한 평가를 위해 evaluation group을 똑같이 해주는것
            
        elif mode == "train":
            self.preprocessed_dataset_wet = self.wav_to_tensor_without_name_and_collect(path = self.path_wet)[0:7500]
            self.preprocessed_dataset_dry = self.wav_to_tensor_without_name_and_collect(path = self.path_dry)
            self.preprocessed_name_list_wet , self.preprocessed_name_list_dry = self.get_sorted_filename_essence_list()
            self.preprocessed_name_list_wet = self.preprocessed_name_list_wet[0:7500]
        elif mode == "valid":
            self.preprocessed_dataset_wet = self.wav_to_tensor_without_name_and_collect(path = self.path_wet)[7501:10000]
            self.preprocessed_dataset_dry = self.wav_to_tensor_without_name_and_collect(path = self.path_dry)
            self.preprocessed_name_list_wet , self.preprocessed_name_list_dry = self.get_sorted_filename_essence_list()
            self.preprocessed_name_list_wet = self.preprocessed_name_list_wet[7501:10000]
        else:
            raise ValueError("give IDMT dataset proper mode")
            
        
        # We don't apply on-the-fly realtime distortion on IDMT dataset. 
        # Distortion is already applied by reference paper authors.
        # So the distorted sample number per one dry source is determined, and the mapping between distorted sample and dry source also determined.
        # Find the dry_source of each dist_audio by matching the filename.
    def dir_of_each_audio_list(self, path) :
        return sorted(glob.glob(path+ "*.wav")) #Must use "Sorted", to conserve order relations between wet_idx, dry_idx, wet_name, dry_name 
    
    def len_before_preprocess(self):
        return len(self.dir_of_each_audio_list())
    
    def __len__(self):
        return self.preprocessed_dataset_wet.shape[0]
    
    def wav_to_tensor_without_name_and_collect(self, path=None):
        import time
        start_tensorizing =time.time()
        tensors_filename = path + f"tensors_order_is_same_as_name.pt"
    
        if os.path.exists(tensors_filename):
            test_tensor_collect = torch.load(tensors_filename)
            print(f"Time Consumed to load {test_tensor_collect.shape[0]}idmt audio tensors: {time.time()-start_tensorizing}")
        
        else:
            dir_of_each_audio_list = self.dir_of_each_audio_list(path)
            for slice_idx, slice_dir in enumerate(tqdm(dir_of_each_audio_list, desc="Tensorize test_dist_audio")):
                slice_np, sr = librosa.load(slice_dir, sr = self.sr, mono = True)
                slice_tensor = torch.tensor(slice_np, dtype=torch.float32).to(self.device).unsqueeze(0)
                if not slice_idx == 0:
                    test_tensor_collect = torch.cat((test_tensor_collect, slice_tensor), dim=0)
                else : #when tensorize first test audio with OD1 or SD1
                    test_tensor_collect = slice_tensor
            torch.save(test_tensor_collect.cpu(), tensors_filename)
            print(f"Time Consumed to convert {test_tensor_collect.shape[0]}idmt wavs to tensors : {time.time()-start_tensorizing}")
            
        if test_tensor_collect.device != "cpu":
            test_tensor_collect = test_tensor_collect.cpu()
        return test_tensor_collect
    
    def get_sorted_filename_essence_list(self):
        wet_name_list_txt = self.path_wet + "audio_name_list.txt"
        
        if os.path.exists(wet_name_list_txt) :
            with open(wet_name_list_txt, "r") as file:
                wet_sources_names_list_essence = [line.strip() for line in file.readlines()]
                # Load list of partial name of audio file. ("/data4/idmt/od1/P64-43110-1111-41185.wav"(X) "P6443110"(O))
        else:
            wet_sources_names_list = self.dir_of_each_audio_list(self.path_wet)
            wet_sources_names_list_essence = [((name.split("/")[-1]).split("-"))[0] + ((name.split("/")[-1]).split("-"))[1] for name in wet_sources_names_list]
            # Make new list of partial each name of audio file. ("/data4/idmt/od1/P64-43110-1111-41185.wav"(X) "P6443110"(O))
            with open(wet_name_list_txt, "w") as file:
                for item in wet_sources_names_list_essence:
                    file.write(f"{item}\n")
            
        dry_name_list_txt = self.path_dry + "audio_name_list.txt"
        if os.path.exists(dry_name_list_txt) :
            with open(dry_name_list_txt, "r") as file:
                dry_sources_names_list_essence = [line.strip() for line in file.readlines()]
                # Load list partial name of audio file. ("P64-43110-1111-41185.wav"(X) "P6443110"(O))
        else:
            dry_sources_names_list = self.dir_of_each_audio_list(self.path_dry)
            dry_sources_names_list_essence = [((name.split("/")[-1]).split("-"))[0]+ ((name.split("/")[-1]).split("-"))[1]for name in dry_sources_names_list]
            # Make new list of partial each name of audio file. ("/data4/idmt/od1/P64-43110-1111-41185.wav"(X) "P6443110"(O))
            with open(dry_name_list_txt, "w") as file:
                for item in dry_sources_names_list_essence:
                    file.write(f"{item}\n")
        return wet_sources_names_list_essence,  dry_sources_names_list_essence

    def __getitem__(self, idx): # load the audio ignoring hexaphonic-property (regard the audio as the superposition of all pickup-cahnnels)
        slice_wet= self.preprocessed_dataset_wet[idx]
        slice_wet_name = self.preprocessed_name_list_wet[idx]
        dry_idx = self.preprocessed_name_list_dry.index(slice_wet_name)
        slice_dry_correspond_to_wet = self.preprocessed_dataset_dry[dry_idx]
        return {"audio_dry_GT" : slice_dry_correspond_to_wet, "audio_wet_GT" : slice_wet ,"wet_idx":idx, "dry_idx":dry_idx}



if __name__ == "__main__":
    def read_yaml_with_dot_parse(yaml_file):
    # Load and parse the YAML file with point(".")
        import omegaconf
        config = omegaconf.OmegaConf.load(yaml_file)
        return config

    config = read_yaml_with_dot_parse("./GDDSP_config_wet.yaml")
    
    dataset_idmt = Test_Dataset_IDMT_Dist(
    config = config,
    device = torch.device("cuda:4"),
    dist = "od1",
    mode = "test"
    ) #dist = "od1" , "808", "mgs"rat"",  #지금 tmux -t 3 에서 맨왼쪽 808, 두번째pane에서 od1 진행 중.
    
    print(dataset_idmt[20]["wet_idx"])
    print(dataset_idmt[20]["dry_idx"])
    import pdb
    pdb.set_trace()
    

    
    

        
    


