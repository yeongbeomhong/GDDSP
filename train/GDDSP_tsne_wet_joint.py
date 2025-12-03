### Draw T-SNE with GutiarSEt Valid Sample 826 excperts + Torch.overdrive 10 / 20 / 30 (826 * 3 = 2478 samples)
# Encode the {pre-Eq 40 bins concat PreAmp1 concat PowerTanh Coefficient 10} = 51-length vector, per audio.
# Check that 51-length vector from overdrive 10 construct cluster together,
# Check that 51-length vector from overdrive 20 construct cluster together,
# Check that 51-length vector from overdrive 30 construct cluster together,

import torch
import torch.nn as nn
import omegaconf
import sys, os, glob
import numpy as np
import argparse
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
import time
from dataset_custom import Dataset_Dist_Slice_Overlap_Realtime
from GDDSP_total_net import GDDSP_wet
import random
from loss.mss_loss import MSSLoss
from optimizer.radam import RAdam

### Training Steps ###
# S0. Load the configurations from config.yaml. 
# Select the GPU as the number which user entered in terminal.(ex> --gpu "2,3") Check whether the device being used is GPU or GPU.
# Enable wandb to trace the log of training loss.
# S1. Declare Dataset and set the data_directory, shuffling, Batch_size, sample_rate, slicing(window size and hop size).
# S2. Load the AutoEncoder (난 지금 Encoder만 불러올거임. Oscillator를 위한 Amp, phase를 뱉는 디코더와 FX chain을 위한 패러미터를 뱉는 디코더 아직안짬)
# S3. Choose loss function, SGD Optimizer, and evaluation metric. #Default = Multi-Scaled Spectral Loss(MSS) and Adam Optimizer
# S4. Load the Learning Schedule and the Validation Period from config file.
# S5. Train.  (Feed the settings of data,net,loss,schedule into trainer)
# S6. Save the tensor of trained Parameters at Checkpoint_folder.
# S7. Save the change in the training loss at log_folder.
### Let's Start ###

# S0. Load the configurations from config.yaml. Select the GPU as the number user enters in terminal. Enable Wandb.
def read_yaml_with_dot_parse(yaml_file):
    # Load and parse the YAML file
    config = omegaconf.OmegaConf.load(yaml_file)
    return config
config = read_yaml_with_dot_parse("./GDDSP_config_wet.yaml")
if config.test_pretrained_model == True:
    config.slice_length = 2
    config.slice_hop =2 
random.seed(config.seed)

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default=None, help= "Select the GPU with terminal argument (e.g. --gpu '3' or --gpu '3,4,5' )")
args = parser.parse_args()

if args.gpu is not None:
    os.environ.update({"CUDA_VISIBLE_DEVICES": str(args.gpu)})
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
    print(f"GPU {str(args.gpu)} will be used during training.")
    print('Count of using GPUs:', torch.cuda.device_count())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Choice of CPU/GPU  :{device}") #Check whether the device being used is GPU or GPU.
else:
    print("No specific GPU was chosen.")
    print('Count of using GPUs:', torch.cuda.device_count())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device :{device}")
    print(f"Choice of CPU/GPU :{device}")  #Check whether the device being used is GPU or GPU.


D=0 # Dataset Option (train dataset for scratch model or pretrained model)
DL=0 # Main Comparison Option : Use W_H "Distortion Layer" =1, Dont Use = 2
AR=0 # Use Random Amount Distortion or Fixed Amount Distortion
RE=0 # Use Reverb or Not
if config.input_dist_type == "torch_overdrive":
    D=1
elif config.input_dist_type == "spotify_pedalboard":
    D=2
    
DL = 1 # joint.py always use W_H Distortion Layer

if config.input_dist_amount_random == True:
    AR=1
else: 
    AR=2
if config.use_room_acoustic == True:
    RE=1
else:
    RE=2
    
if config.input_dist_amount_random == True:
    experiment_full_name =  config.experiment_name + \
        f"Test_D{D}DL{DL}AR{AR}RE{RE}_nHarm{config.n_harmonics}_PreampCtanhSig_True_joint_group_f0silences"
else:
    experiment_full_name =  config.experiment_name + \
        f"Test_D{D}DL{DL}AR{AR}RE{RE}__FixedDist{config.input_dist_amount_fixed}_nHarm{config.n_harmonics}_PreampCtanhSig_True_joint_group_f0silences"

# S1. Load Dataset and set the configurations of data_directory, shuffling, Batch_size, sample_rate, slicing(window size and hop size).

config_fixed_dist_10 = read_yaml_with_dot_parse("./GDDSP_config_tsne_1.yaml") # Not Config for Network Layer Construction. Config only for dataset
# config_fixed_dist_20 = read_yaml_with_dot_parse("./GDDSP_config_tsne_2.yaml")
config_fixed_dist_30 = read_yaml_with_dot_parse("./GDDSP_config_tsne_3.yaml")

valid_dataset_10 = Dataset_Dist_Slice_Overlap_Realtime(
    config = config_fixed_dist_10,
    device = device,
    audio_path = config.train_path
)
'''
valid_dataset_20 = Dataset_Dist_Slice_Overlap_Realtime(
    config = config_fixed_dist_20,
    device = device,
    audio_path = config.valid_path
)
'''
valid_dataset_30 =  Dataset_Dist_Slice_Overlap_Realtime(
    config = config_fixed_dist_30,
    device = device,
    audio_path = config.train_path
)

valid_dataloader_10 = DataLoader(
    valid_dataset_10 ,
    batch_size = config.batch_size,
    shuffle=False,
    num_workers=config.num_workers,
    pin_memory=False,
)
'''
valid_dataloader_20 = DataLoader(
    valid_dataset_20 ,
    batch_size = config.batch_size,
    shuffle=False,
    num_workers=config.num_workers,
    pin_memory=False,
)
'''
valid_dataloader_30 = DataLoader(
    valid_dataset_30 ,
    batch_size = config.batch_size,
    shuffle=False,
    num_workers=config.num_workers,
    pin_memory=False,
)



# S2. Load the Total Strudcture of GDDSP (= encoder- decoder - osillator and noise generator - FX chain)
net = GDDSP_wet(config)
if config.test_check_point_load_path == None :
    raise ValueError("You must give pre-trained checkpoint for TSNE.")
else: # load parameters of checkpoint
    print("You chose to test the model of checkpoint")
    print(f"The path of checkpoint is {config.test_check_point_load_path}")
    pretrained_state_dict = torch.load(config.test_check_point_load_path, map_location = torch.device('cpu'))
    net.load_state_dict(pretrained_state_dict["model_state_dict"])
    start_epoch = pretrained_state_dict["epoch"] # succeeding the finished epochs of pre-trained ckp.
    print("The parameters has been loaded.")
net.to(device)


# S3. Choose test loss metric, and signal_key for picking audio from Recon_audio_dictionary which will be compared with input GT
if config.use_dist == True:
    if config.use_room_acoustic == True:
        signal_key = "audio_dist_room"
        de_distortion_signal_key = "audio_dry"
    else:
        signal_key = "audio_dist"
        de_distortion_signal_key = "audio_dry"
        
print("The signal_key to be compared with input dist audio is ", signal_key)
loss = MSSLoss([2048, 1024, 512, 256], signal_key = signal_key).to(device)

# --------------------------------------- #
trainable_parameters_list = filter(lambda p: p.requires_grad, net.parameters())
num_trainable_parameters = sum([torch.numel(p) for p in trainable_parameters_list])
print("Num of trainable parameters in PRE-TRAINED NET : ", num_trainable_parameters)

## Main test dataset_a TEST Loop. It's not training, so dont need multi epochs ##
time_start_test= time.time()
print(f"Start Gather preEQ-preAmp-Ctanh Vector with torch drive 10 dB data")


with torch.no_grad(): # Test doesnt need grad graph
    for batch_idx, batch in tqdm(enumerate(valid_dataloader_10), desc="Batches") : # Iterations in one epoch
        print(f"Now processing {batch_idx}th/{len(valid_dataloader_10)} in 10dB GAIN batch")
        batch["audio_wet_GT"] = batch["audio_wet_GT"].to(device) #This is input test audio (distorted)        
        knobs = net.produce_distortion_knobs(batch)
        if batch_idx == 0 :
            knobs_tensor_collect = knobs
        else :
            knobs_tensor_collect = torch.cat((knobs_tensor_collect, knobs), dim = 0)
    '''
    for batch_idx, batch in tqdm(enumerate(valid_dataloader_20), desc="Batches") :
        print(f"Now processing {batch_idx}th/{len(valid_dataloader_20)} in 20dB GAIN batch")
        batch["audio_wet_GT"] = batch["audio_wet_GT"].to(device) #This is input test audio (distorted)        
        knobs = net.produce_distortion_knobs(batch)
        knobs_tensor_collect = torch.cat((knobs_tensor_collect, knobs), dim = 0)
    '''
        
    for batch_idx, batch in tqdm(enumerate(valid_dataloader_30), desc="Batches") :
        print(f"Now processing {batch_idx}th/{len(valid_dataloader_30)} in 30dB GAIN batch")
        batch["audio_wet_GT"] = batch["audio_wet_GT"].to(device) #This is input test audio (distorted)        
        knobs = net.produce_distortion_knobs(batch)
        knobs_tensor_collect = torch.cat((knobs_tensor_collect, knobs), dim = 0)
    
print("Distortion Knobs Shape collected from torch overdrive 10dB/30dB + valid Guitarset :" ,  knobs_tensor_collect.shape)
knobs_list_for_tsne = knobs_tensor_collect.cpu().numpy()
gain_labels_10 = np.full(len(valid_dataset_10), 10)
'''
gain_labels_20 = np.full(len(valid_dataset_20), 20)
'''
gain_labels_30 = np.full(len(valid_dataset_30), 30)
labels_list_for_tsne =  np.concatenate((gain_labels_10, gain_labels_30))
print("Overdrive Gain(dB) GT labes are constructed. shape is : ", labels_list_for_tsne.shape)
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
tsne = TSNE(n_components=2)
knobs_list_2D_visualized = tsne.fit_transform(knobs_list_for_tsne)

# Create a scatter plot
plt.figure(figsize=(10, 8))
scatter = plt.scatter(knobs_list_2D_visualized[:, 0], knobs_list_2D_visualized[:, 1], c=labels_list_for_tsne, cmap='viridis')

# Add labels and legend
plt.title('t-SNE of Distortion Knobs = [pre-EQ, pre-Amp, Tanh_coefficents] reduced to 2D space')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend(*scatter.legend_elements(), title='overdrive dB')

plt.savefig('./tsne/tsne_plot_train_set.png')