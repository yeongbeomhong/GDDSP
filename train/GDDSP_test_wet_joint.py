### This is the main script of GDDSP that utilizes all sub-scripts for training ###

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
from dataset_custom import Test_Dataset_IDMT_Dist
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

import wandb

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
        f"Test_D{D}DL{DL}AR{AR}RE{RE}_nHarm{config.n_harmonics}_finalthesis_joint_trainGuitarSet_testIDMT"
else:
    experiment_full_name =  config.experiment_name + \
        f"Test_D{D}DL{DL}AR{AR}RE{RE}__nHarm{config.n_harmonics}_finalthesis_joint_trainGuitarSet_testIDMT"

wandb.init(project="GDDSP_Test", name= experiment_full_name)

# S1. Load Dataset and set the configurations of data_directory, shuffling, Batch_size, sample_rate, slicing(window size and hop size).
dist_a = "od1"
dist_b = "808"
dist_c = "mgs"
dist_d = "rat"
test_dataset_a = Test_Dataset_IDMT_Dist(
    config = config,
    device = device,
    dist = dist_a,
    mode = "test"
)
test_dataset_b = Test_Dataset_IDMT_Dist(
    config = config,
    device = device,
    dist = dist_b,
    mode = "test"
)
test_dataset_c = test_dataset_b = Test_Dataset_IDMT_Dist(
    config = config,
    device = device,
    dist = dist_c,
    mode = "test"
)
test_dataset_d = test_dataset_b = Test_Dataset_IDMT_Dist(
    config = config,
    device = device,
    dist = dist_d,
    mode = "test"
)

test_dataloader_a = DataLoader(
    test_dataset_a,
    batch_size = config.batch_size,
    shuffle=False,
    num_workers=config.num_workers,
    pin_memory=False,
)
test_dataloader_b = DataLoader(
    test_dataset_b,
    batch_size = config.batch_size,
    shuffle=False,
    num_workers=config.num_workers,
    pin_memory=False,
)
test_dataloader_c = DataLoader(
    test_dataset_c,
    batch_size = config.batch_size,
    shuffle=False,
    num_workers=config.num_workers,
    pin_memory=False,
)
test_dataloader_d = DataLoader(
    test_dataset_d,
    batch_size = config.batch_size,
    shuffle=False,
    num_workers=config.num_workers,
    pin_memory=False,
)


print(f"Two test_dataloader with Real-World Pedal, {dist_a} and {dist_b} has been built.")
print(f"Counts of test audio-{dist_a} exceprts with 2-sec: {len(test_dataset_a)}")
print(f"Counts of test audio-{dist_b} exceprts with 2-sec: {len(test_dataset_b)}")

# S2. Load the Total Strudcture of GDDSP (= encoder- decoder - osillator and noise generator - FX chain)
net = GDDSP_wet(config)
if config.test_check_point_load_path == None :
    raise ValueError("You must give pre-trained checkpoint for test eval.")
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
print(f"Start Test Iterations with {dist_a} distortion data")
avg_dist_recon_loss_a = 0 
with torch.no_grad(): # Test doesnt need grad graph
    for batch_idx, batch in tqdm(enumerate(test_dataloader_a), desc="Batches") : # Iterations in one epoch
        start_iter =time.time()
        print(f"Now processing {batch_idx}th/{len(test_dataloader_a)} batch")
        print("Counts of 2_sec_sliced audio in one test batch : ", batch["audio_wet_GT"].shape[0])

        batch["audio_wet_GT"] = batch["audio_wet_GT"].to(device) #This is input test audio (distorted)
        audio_recon = net(batch)
        dist_recon_loss_a = loss(audio_recon[signal_key], batch["audio_wet_GT"]).item()
        #audio_recon[signal_key] = audio_dist_room if pre-trained joint GDDSP used distortion layer and reverb during training
        # = audio_dist if pre-trained GDDSP model used only distortion layer during training
        avg_dist_recon_loss_a += dist_recon_loss_a
        print("Time Consuming for One TEST Iteration is : ", time.time()-start_iter)
        
    avg_dist_recon_loss_a = avg_dist_recon_loss_a / len(test_dataloader_a)
    wandb.log({f"(joint, {dist_a}) wet_test_loss": avg_dist_recon_loss_a,
            "num_trainable_params" : num_trainable_parameters})
    
## save the sample test audio in dataset_a reconstructed by GDDSP periodically with .wav format, and compare it with original audio of test dataset
if not os.path.exists(config.sample_audio_recon_save_path + experiment_full_name):
    os.makedirs(config.sample_audio_recon_save_path + experiment_full_name)
audio_path = config.sample_audio_recon_save_path + experiment_full_name
sample_original_test_audio_a_1 = batch["audio_wet_GT"][0].detach().cpu().numpy()
sample_original_test_audio_a_2 = batch["audio_wet_GT"][1].detach().cpu().numpy()
sample_recon_test_audio_a_1 = audio_recon[signal_key][0].detach().cpu().numpy()
sample_recon_test_audio_a_2 = audio_recon[signal_key][1].detach().cpu().numpy()

sample_recon_test_audio_a_1_de_distortion = audio_recon[de_distortion_signal_key][0].detach().cpu().numpy()
sample_recon_test_audio_a_2_de_distortion = audio_recon[de_distortion_signal_key][1].detach().cpu().numpy()

import soundfile as sf
sr = config.sample_rate
sf.write(audio_path + f"/original_{dist_a}_1.wav", sample_original_test_audio_a_1, sr)
sf.write(audio_path + f"/original_{dist_a}_2.wav", sample_original_test_audio_a_2, sr)
sf.write(audio_path + f"/recon_{dist_a}_1.wav", sample_recon_test_audio_a_1, sr)
sf.write(audio_path + f"/recon_{dist_a}_2.wav", sample_recon_test_audio_a_2, sr)
sf.write(audio_path + f"/recon_{dist_a}_dedist_1.wav", sample_recon_test_audio_a_1_de_distortion, sr)
sf.write(audio_path + f"/recon_{dist_a}_dedist_2.wav", sample_recon_test_audio_a_2_de_distortion, sr)
print(f"Sample Test audio_{dist_a} reconstructed by GDDSP is saved at {audio_path}")

## Main test dataset_b TEST Loop. It's not training, so dont need multi epochs ##
print(f"Start Test Iterations with {dist_b} distortion data")
avg_dist_recon_loss_b = 0 
with torch.no_grad(): # Test doesnt need grad graph
    for batch_idx, batch in tqdm(enumerate(test_dataloader_b), desc="Batches") : # Iterations in one epoch
        start_iter =time.time()
        print(f"Now processing {batch_idx}th/{len(test_dataloader_b)} batch")
        print("Counts of 2_sec_sliced audio in one test batch : ", batch["audio_wet_GT"].shape[0])

        batch["audio_wet_GT"] = batch["audio_wet_GT"].to(device) #This is input test audio (distorted)
        audio_recon = net(batch)
        dist_recon_loss_b = loss(audio_recon[signal_key], batch["audio_wet_GT"]).item()
        #audio_recon[signal_key] = audio_dist_room if pre-trained joint GDDSP used distortion layer and reverb during training
        # = audio_dist if pre-trained GDDSP model used only distortion layer during training
        avg_dist_recon_loss_b += dist_recon_loss_b
        print("Time Consuming for One TEST Iteration is : ", time.time()-start_iter)
        
    avg_dist_recon_loss_b = avg_dist_recon_loss_b / len(test_dataloader_b)
    wandb.log({f"(joint,{dist_b}) wet_test_loss": avg_dist_recon_loss_b,
            "num_trainable_params" : num_trainable_parameters})
    
## save the sample test audio in dataset_b reconstructed by GDDSP periodically with .wav format, and compare it with original audio of test dataset
audio_path = config.sample_audio_recon_save_path + experiment_full_name
sample_original_test_audio_b_1 = batch["audio_wet_GT"][0].detach().cpu().numpy()
sample_original_test_audio_b_2 = batch["audio_wet_GT"][1].detach().cpu().numpy()
sample_recon_test_audio_b_1 = audio_recon[signal_key][0].detach().cpu().numpy()
sample_recon_test_audio_b_2 = audio_recon[signal_key][1].detach().cpu().numpy()

sample_recon_test_audio_b_1_de_distortion = audio_recon[de_distortion_signal_key][0].detach().cpu().numpy()
sample_recon_test_audio_b_2_de_distortion = audio_recon[de_distortion_signal_key][1].detach().cpu().numpy()

import soundfile as sf
sr = config.sample_rate
sf.write(audio_path + f"/original_{dist_b}_1.wav", sample_original_test_audio_a_1, sr)
sf.write(audio_path + f"/original_{dist_b}_2.wav", sample_original_test_audio_a_2, sr)
sf.write(audio_path + f"/recon_{dist_b}_1.wav", sample_recon_test_audio_a_1, sr)
sf.write(audio_path + f"/recon_{dist_b}_2.wav", sample_recon_test_audio_a_2, sr)
sf.write(audio_path + f"/recon_{dist_b}_dedist_1.wav", sample_recon_test_audio_a_1_de_distortion, sr)
sf.write(audio_path + f"/recon_{dist_b}_dedist_2.wav", sample_recon_test_audio_a_2_de_distortion, sr)
print(f"Sample Test audio_{dist_b} reconstructed by GDDSP is saved at {audio_path}")


## Main test dataset_c TEST Loop. It's not training, so dont need multi epochs ##
print(f"Start Test Iterations with {dist_c} distortion data")
avg_dist_recon_loss_c = 0 
with torch.no_grad(): # Test doesnt need grad graph
    for batch_idx, batch in tqdm(enumerate(test_dataloader_c), desc="Batches") : # Iterations in one epoch
        start_iter =time.time()
        print(f"Now processing {batch_idx}th/{len(test_dataloader_c)} batch")
        print("Counts of 2_sec_sliced audio in one test batch : ", batch["audio_wet_GT"].shape[0])

        batch["audio_wet_GT"] = batch["audio_wet_GT"].to(device) #This is input test audio (distorted)
        audio_recon = net(batch)
        dist_recon_loss_c = loss(audio_recon[signal_key], batch["audio_wet_GT"]).item()
        #audio_recon[signal_key] = audio_dist_room if pre-trained joint GDDSP used distortion layer and reverb during training
        # = audio_dist if pre-trained GDDSP model used only distortion layer during training
        avg_dist_recon_loss_c += dist_recon_loss_c
        print("Time Consuming for One TEST Iteration is : ", time.time()-start_iter)
        
    avg_dist_recon_loss_c = avg_dist_recon_loss_c / len(test_dataloader_c)
    wandb.log({f"(joint,{dist_c}) wet_test_loss": avg_dist_recon_loss_c,
            "num_trainable_params" : num_trainable_parameters})
    
## save the sample test audio in dataset_c reconstructed by GDDSP periodically with .wav format, and compare it with original audio of test dataset
audio_path = config.sample_audio_recon_save_path + experiment_full_name
sample_original_test_audio_c_1 = batch["audio_wet_GT"][0].detach().cpu().numpy()
sample_original_test_audio_c_2 = batch["audio_wet_GT"][1].detach().cpu().numpy()
sample_recon_test_audio_c_1 = audio_recon[signal_key][0].detach().cpu().numpy()
sample_recon_test_audio_c_2 = audio_recon[signal_key][1].detach().cpu().numpy()

sample_recon_test_audio_c_1_de_distortion = audio_recon[de_distortion_signal_key][0].detach().cpu().numpy()
sample_recon_test_audio_c_2_de_distortion = audio_recon[de_distortion_signal_key][1].detach().cpu().numpy()

import soundfile as sf
sr = config.sample_rate
sf.write(audio_path + f"/original_{dist_c}_1.wav", sample_original_test_audio_c_1, sr)
sf.write(audio_path + f"/original_{dist_c}_2.wav", sample_original_test_audio_c_2, sr)
sf.write(audio_path + f"/recon_{dist_c}_1.wav", sample_recon_test_audio_c_1, sr)
sf.write(audio_path + f"/recon_{dist_c}_2.wav", sample_recon_test_audio_c_2, sr)
sf.write(audio_path + f"/recon_{dist_c}_dedist_1.wav", sample_recon_test_audio_c_1_de_distortion, sr)
sf.write(audio_path + f"/recon_{dist_c}_dedist_2.wav", sample_recon_test_audio_c_2_de_distortion, sr)
print(f"Sample Test audio_{dist_c} reconstructed by GDDSP is saved at {audio_path}")


## Main test dataset_d TEST Loop. It's not training, so dont need multi epochs ##
print(f"Start Test Iterations with {dist_d} distortion data")
avg_dist_recon_loss_d = 0 
with torch.no_grad(): # Test doesnt need grad graph
    for batch_idx, batch in tqdm(enumerate(test_dataloader_d), desc="Batches") : # Iterations in one epoch
        start_iter =time.time()
        print(f"Now processing {batch_idx}th/{len(test_dataloader_d)} batch")
        print("Counts of 2_sec_sliced audio in one test batch : ", batch["audio_wet_GT"].shape[0])

        batch["audio_wet_GT"] = batch["audio_wet_GT"].to(device) #This is input test audio (distorted)
        audio_recon = net(batch)
        dist_recon_loss_d = loss(audio_recon[signal_key], batch["audio_wet_GT"]).item()
        #audio_recon[signal_key] = audio_dist_room if pre-trained joint GDDSP used distortion layer and reverb during training
        # = audio_dist if pre-trained GDDSP model used only distortion layer during training
        avg_dist_recon_loss_d += dist_recon_loss_d
        print("Time Consuming for One TEST Iteration is : ", time.time()-start_iter)
        
    avg_dist_recon_loss_d = avg_dist_recon_loss_d / len(test_dataloader_d)
    wandb.log({f"(joint,{dist_d}) wet_test_loss": avg_dist_recon_loss_d,
            "num_trainable_params" : num_trainable_parameters})
    
## save the sample test audio in dataset_c reconstructed by GDDSP periodically with .wav format, and compare it with original audio of test dataset
audio_path = config.sample_audio_recon_save_path + experiment_full_name
sample_original_test_audio_d_1 = batch["audio_wet_GT"][0].detach().cpu().numpy()
sample_original_test_audio_d_2 = batch["audio_wet_GT"][1].detach().cpu().numpy()
sample_recon_test_audio_d_1 = audio_recon[signal_key][0].detach().cpu().numpy()
sample_recon_test_audio_d_2 = audio_recon[signal_key][1].detach().cpu().numpy()

sample_recon_test_audio_d_1_de_distortion = audio_recon[de_distortion_signal_key][0].detach().cpu().numpy()
sample_recon_test_audio_d_2_de_distortion = audio_recon[de_distortion_signal_key][1].detach().cpu().numpy()

import soundfile as sf
sr = config.sample_rate
sf.write(audio_path + f"/original_{dist_d}_1.wav", sample_original_test_audio_d_1, sr)
sf.write(audio_path + f"/original_{dist_d}_2.wav", sample_original_test_audio_d_2, sr)
sf.write(audio_path + f"/recon_{dist_d}_1.wav", sample_recon_test_audio_d_1, sr)
sf.write(audio_path + f"/recon_{dist_d}_2.wav", sample_recon_test_audio_d_2, sr)
sf.write(audio_path + f"/recon_{dist_d}_dedist_1.wav", sample_recon_test_audio_d_1_de_distortion, sr)
sf.write(audio_path + f"/recon_{dist_d}_dedist_2.wav", sample_recon_test_audio_d_2_de_distortion, sr)
print(f"Sample Test audio_{dist_d} reconstructed by GDDSP is saved at {audio_path}")