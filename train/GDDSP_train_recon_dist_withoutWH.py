### This is the main script of GDDSP that utilizes all sub-scripts for training ###
# This Script does not use Joint hybrid loss. There is no intermediate loss. Only One dist recon loss will be calculated with output of reverb or HO
# Dont Use 2 losses for SGD, But will save sample recon audio with 2step (= output of H.O+N.G, and output after reverb)
# This saving manner aims to figure out the Actual Role of Reverb Module. (Does it find out distortion? or find out reverb?)
import time
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

from dataset_custom import  Dataset_Dist_Slice_Overlap_Realtime, Test_Dataset_IDMT_Dist
from GDDSP_total_net import GDDSP_wet
import random
from loss.mss_loss import MSSLoss
from optimizer.radam import RAdam


### Training Steps ###
# S0. Load the configurations from config.yaml. Set the Project name to be recorded in Wandb 
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

config = read_yaml_with_dot_parse("./GDDSP_config_recon_dist_withoutWH.yaml")
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

D=0 # Dataset Option
DL=0 # Main Comparison Option : Use W_H "Distortion Layer" or not
AR=0 # Use Random Amount Distortion or Fixed Amount Distortion
RE=0 # Use Reverb or Not
if config.input_dist_type == "torch_overdrive":
    D=1
elif config.input_dist_type == "spotify_pedalboard":
    D=2
elif config.input_dist_type == "guitar_real_plugin":
    D=3

if config.n_harmonics <= 101 :
    DL = 2 #NN (Neither DL, Nor More Harmonics) recon_dist_without_fx_chain.py always dont use W_H Distortion Layer
elif config.n_harmonics > 101 :
    DL = 3 #MH (More Harmoncis)
else:
    pass

if config.input_dist_amount_random == True:
    AR=1
else: 
    AR=2
if config.use_room_acoustic == True:
    RE=1
else:
    RE=2
    
import wandb
if config.input_dist_amount_random == True :
    experiment_full_name =  config.experiment_name + \
        f"D{D}DL{DL}AR{AR}RE{RE}_nHarm{config.n_harmonics}_ReconDist_withoutWH_no_2step_loss_trainIDMT{config.train_and_valid_with_idmt}{config.idmt_dist_type}"
else:
    experiment_full_name =  config.experiment_name + \
        f"D{D}DL{DL}AR{AR}RE{RE}_nHarm{config.n_harmonics}_ReconDist_withoutWH_no_2step_loss_trainIDMT{config.train_and_valid_with_idmt}{config.idmt_dist_type}"
    
wandb.init(project="GDDSP_Train", name= experiment_full_name)

# S1. Load Dataset and set the configurations of data_directory, shuffling, Batch_size, sample_rate, slicing(window size and hop size).
if config.train_and_valid_with_idmt : 
    # Split IDMT wet dist audio recorded by 4 players into 3:1, train with 3 and validation with 1 (7500:2500)
    # Check that WHdist layer outperform 141harmonics, when experience the target distortion plugin in Training Phase
    train_dataset = Test_Dataset_IDMT_Dist(
        config = config,
        device = device, 
        dist = config.idmt_dist_type, 
        mode = "train"
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=False,
    )
    valid_dataset = Test_Dataset_IDMT_Dist(
        config = config,
        device = device, 
        dist = config.idmt_dist_type, 
        mode = "valid"
    )
    valid_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=False,
    )
else:
    train_dataset = Dataset_Dist_Slice_Overlap_Realtime( #Construct Train dataset with guitarset and torch.overdrive library
        config = config,
        device = device,
        audio_path = config.train_path
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=False,
    )

    valid_dataset = Dataset_Dist_Slice_Overlap_Realtime(
        config = config,
        device = device,
        audio_path = config.valid_path
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size = config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=False,
    )


print("The train_dataset, valid_dataset and dataloader has been built.")
print(f"Total Counts of training audio excerpts with fixed duration : {len(train_dataset)}")
print(f"Total Time_durtaion of training audio excerpts : {config.slice_length * len(train_dataset)}")

# S2. Load the Total Strudcture of GDDSP (= encoder- decoder - osillator and noise generator - FX chain)
net = GDDSP_wet(config)

if config.train_check_point_load_path == None :
    print("The New Model will be trained from scratch_state.(Randomly Initialized Parameters)")
    start_epoch = 0
else: # load parameters of checkpoint
    print("You chose to train the model from checkpoint.(Continuing the previous training)")
    print(f"The path of checkpoint is {config.train_check_point_load_path}")
    pretrained_state_dict = torch.load(config.train_check_point_load_pathh)
    net.load_state_dict(pretrained_state_dict["model_state_dict"])
    start_epoch = pretrained_state_dict["epoch"] # succeeding the finished epochs of pre-trained ckp.
    print("The parameters has been loaded.")
net.to(device)


# S3. Choose loss function, SGD Optimizer, and validation metric. #Default = Multi-Scaled Spectral Loss(MSS) and Adam Optimizer
# Define training loss, and select signal key of input to be calculated in loss
if config.use_room_acoustic == True:
    signal_key = "audio_room"
else:
    signal_key = "audio_dry"
print("The signal_key to be compared with input dist audio is ", signal_key)
loss = MSSLoss([2048, 1024, 512, 256], signal_key = signal_key).to(device)

# Define evaluation(validation) metrics
if config.metric == "mss":
    def metric(output, gt):
        with torch.no_grad(): #no grad_backpropagtion during validation
            return loss(output, gt)
        
elif config.metric == "f0":
    # TODO Implement
    raise NotImplementedError
else:
    raise NotImplementedError


# Setting Optimizer and Gradient Limit Clipping
if config.optimizer == "adam":
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, net.parameters()), lr=config.lr)
elif config.optimizer == "radam":
    optimizer = RAdam(filter(lambda x: x.requires_grad, net.parameters()), lr=config.lr)
else:
    raise NotImplementedError
max_norm = 5.

# S4. Load the Learning Schedule and the Validation Period from config file.
# Setting Scheduler
if config.lr_scheduler == "cosine":
    # restart every T_0 * validation_interval steps
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, eta_min=config.lr_min
    ) # The input argument "optimizer" to define the scheduler must matches with your optimizer's name
     
elif config.lr_scheduler == "plateau":
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=config.lr_decay
    )
elif config.lr_scheduler == "multi":
    # learning rate decreases by every 50 epochs
    # 1~50 epoch :     lr = config.lr
    # 51~100 epoch :   lr =  config.lr * gamma
    # 101~150 epoch :   lr =  config.lr * gamma * gamma 
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        [(x + 1) * config.lr_decay_period for x in range(50)], # epoch turning point to decrease lr : [10 ,20, ... 500]
        gamma=config.lr_decay, #lr decreasing ratio
    )
    # Write scheduler.step() in Epoch-Loop.
    # Then the scheduler will count the finished epochs
    # When the counts of finished epochs match 10, 20 ,.. 500, The scheduler reduces lr of optimizer with decay ratio.
    print(f"The lr_scheduler will reduce the learning rate every {config.lr_decay_period} epoch.")
    
elif config.lr_scheduler == "no":
    scheduler = None
else:
    raise ValueError(f"unknown lr_scheduler : {config.lr_scheduler}")

print(f"The learning Schedule during training : {config.lr_scheduler}")
# --------------------------------------- #


# Main Training Loop (zero_grad, forward propagation with network, backpropagation, save parameters at ckp) #'
trainable_parameters_list = filter(lambda p: p.requires_grad, net.parameters())
num_trainable_parameters = sum([torch.numel(p) for p in trainable_parameters_list])
print("\n The num of trainable parameters in net : ", num_trainable_parameters)
time_start_train = time.time()
for epoch in tqdm(range(config.max_epoch), desc="epochs"):
    epoch += 1
    avg_loss_value = 0
    print(f"Disabled FX_Decoder and W_H_Distortion. Now GDDSP reconstrucsts Distorted audio only with sine fundamentals, {config.n_harmonics}harmonics, and noises")
    print(f"There are {len(train_dataloader)} batches per epoch, and {config.batch_size} audio excerpts per batch.")
    print(f"Now Start the training of {start_epoch + epoch}th epoch.")
    for batch_idx, batch in tqdm(enumerate(train_dataloader), desc="Batches") : # Iterations in one epoch
        # Feed Input audio to GPU (batch-wise)
        start_iter = time.time()
        
        batch["audio_wet_GT"] = batch["audio_wet_GT"].to(device)
        print(f"Now processing {batch_idx}th/{len(train_dataloader)} batch in {start_epoch + epoch}th epoch.")
        audio_recon = net(batch)
        loss_value = loss(audio_recon[signal_key], batch["audio_wet_GT"])
        # audio_recon[signal_key] = Sum of Outputs of Oscillators and Noise Generators + (optional)apply global reverb or not
        # Penalize this output to be close to Distorted_Input_GT ("Recon dist input only with dry synthesizer")
        optimizer.zero_grad() # Clear the gradients         
        loss_value.backward()  # Compute gradients
        nn.utils.clip_grad_norm_(net.parameters(), max_norm)
        optimizer.step()  # Update model parameters
        avg_loss_value += loss_value.detach().item() # avg_loss = average loss of all audio data in one epoch
        print("(Wihtout W_H Dist) Time Consuming for One Iteration is : ", time.time()-start_iter)
    scheduler.step()
    avg_loss_value = avg_loss_value / len(train_dataloader)
    wandb.log({"(Without W_H Dist)loss between Oscillator_Recon and Distorted_Input": avg_loss_value,
               "epoch" : start_epoch+epoch,
               "elapsed_time": time.time() - time_start_train,
               "num_trainable_params" : num_trainable_parameters})
    
    
    ## save the parameters periodically
    if epoch % config.check_point_save_period == 0 and epoch > 100 :
        if not os.path.exists(config.check_point_save_path + experiment_full_name):
            os.makedirs(config.check_point_save_path + experiment_full_name)
        ckp_path = config.check_point_save_path + experiment_full_name + f"/without_fx_layers_epoch{start_epoch + epoch}.pt"
        torch.save({
            'epoch': start_epoch + epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss_value
            }, ckp_path)
        print(f"The trained parameters are saved at {ckp_path}")
    
    ## save the sample audio reconstructed by GDDSP periodically with .wav format, and compare it with original audio of same data_idx.
    if epoch % config.sample_audio_recon_save_period == 0 and epoch > 100 :
        import soundfile as sf
        sr = config.sample_rate
        if not os.path.exists(config.sample_audio_recon_save_path + experiment_full_name):
            os.makedirs(config.sample_audio_recon_save_path + experiment_full_name)
        audio_path = config.sample_audio_recon_save_path + experiment_full_name + f"/NoWH_epoch{start_epoch + epoch}"
        sample_original_audio_dist = batch["audio_wet_GT"][0].detach().cpu().numpy()
        sample_original_audio_dry = batch["audio_dry_GT"][0] # No need to detach, It never come into encoder or compared in loss function.
        sf.write(audio_path + "_original_dist.wav", sample_original_audio_dist, sr)
        sf.write(audio_path + "_original_dry.wav", sample_original_audio_dry, sr)

        if config.use_room_acoustic == True:
            sample_recon_audio_before_reverb = audio_recon["audio_dry"][0].detach().cpu().numpy() #audio_dry
            sample_recon_audio_after_reverb = audio_recon["audio_room"][0].detach().cpu().numpy() #audio_room
            sf.write(audio_path + "_recon_dist_with_HO+NG+RE_dereverb.wav", sample_recon_audio_before_reverb, sr)
            sf.write(audio_path + "_recon_dist_with_HO+NG+RE.wav", sample_recon_audio_after_reverb, sr)
        else :
            sample_recon_audio_before_reverb = audio_recon["audio_dry"][0].detach().cpu().numpy() #audio_dry
            sf.write(audio_path + "_recon_dist_with_HO+NG.wav", sample_recon_audio_before_reverb, sr)
        print(f"The sample audio reconstructed by GDDSP is saved at {audio_path}")
    
    ## Calculate valid loss Periodically, and Trace Valid loss with wandb
    if epoch % config.valid_period  == 0:
        avg_valid_loss_dry = 0
        avg_valid_loss_wet = 0
        avg_valid_loss_value = 0
        optimizer.zero_grad() # Clear the gradients
        print(f"validation at {start_epoch + epoch}th epoch started.")
        with torch.no_grad():
            for batch_idx, valid_batch in tqdm(enumerate(valid_dataloader), desc="Batches"):
                valid_batch["audio_wet_GT"] = valid_batch["audio_wet_GT"].to(device)
                audio_recon = net(valid_batch)
                valid_loss_value = loss(audio_recon[signal_key], valid_batch["audio_wet_GT"]) #recon[audio_dry] = Sum of outputs of Oscillators and the Noise Generator.
                avg_valid_loss_value += valid_loss_value.detach().item()
        avg_valid_loss_value = avg_valid_loss_value / len(valid_dataloader)
        wandb.log({"(Without W_H Dist) Valid loss between Oscillator_Recon and Distorted_Input": avg_valid_loss_value,
                   "epoch" : start_epoch+epoch})
        print(f"validation at {start_epoch + epoch}th epoch finished.")
