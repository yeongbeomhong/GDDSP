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
from dataset_custom import Dataset_Dist_Slice_Overlap_Realtime, Test_Dataset_IDMT_Dist
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
random.seed(config.seed)
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default=None, help= "Select the GPU with terminal argument (e.g. --gpu '3' or --gpu '3,4,5' )")
args = parser.parse_args()

if args.gpu != "cpu" and args.gpu != None:
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
        f"D{D}DL{DL}AR{AR}RE{RE}_nHarm{config.n_harmonics}_nTanh{config.distortion.n_tanh}_zfx{config.zfx_units}_joint_(halfZFXunitZFXmlpCtanhPreEQ)_trainIDMT{config.train_and_valid_with_idmt}{config.idmt_dist_type}"
else:
    experiment_full_name =  config.experiment_name + \
        f"D{D}DL{DL}AR{AR}RE{RE}_nHarm{config.n_harmonics}_nTanh{config.distortion.n_tanh}_zfx{config.zfx_units}_joint_(halfZFXunitZFXmlpCtanhPreEQ)_trainIDMT{config.train_and_valid_with_idmt}{config.idmt_dist_type}"

# 패러미터 수 세느라 잠깐 이쪽으로 올려둠.
# S2. Load the Total Strudcture of GDDSP (= encoder- decoder - osillator and noise generator - FX chain)
net = GDDSP_wet(config)


if config.train_check_point_load_path == None :
    print("The New Model will be trained from scratch_state.(Randomly Initialized Parameters)")
    start_epoch = 0
else: # load parameters of checkpoint
    print("You chose to train the model from checkpoint.(Continuing the previous training)")
    print(f"The path of checkpoint is {config.train_check_point_load_path}")
    pretrained_state_dict = torch.load(config.train_check_point_load_path)
    net.load_state_dict(pretrained_state_dict["model_state_dict"])
    start_epoch = pretrained_state_dict["epoch"] # succeeding the finished epochs of pre-trained ckp.
    print("The parameters has been loaded.")

#Parellel train code
if torch.cuda.device_count() > 1: # Parallel training to efficeintly use Multiple GPUs' memories
    print(f"Use {torch.cuda.device_count()} GPUs for Multi-GPU-Parallel-Training.")
    net = nn.DataParallel(net)
else :
    pass
net.to(device)


## GET Num of parameters ##
def get_num_param(net):
    for name, param in net.named_parameters():
        if param.requires_grad:
            print(f"Trainable Layer: {name}, Parameters: {param.numel()}")
        else:
            print(f"Non-Trainable Layer: {name}, Parameters: {param.numel()}")
    return None
get_num_param(net)

import pdb
pdb.set_trace()


wandb.init(project="GDDSP_Train", name= experiment_full_name)



# S1. Load Dataset and set the configurations of data_directory, shuffling, Batch_size, sample_rate, slicing(window size and hop size).

if config.train_and_valid_with_idmt: 
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
    train_dataset = Dataset_Dist_Slice_Overlap_Realtime( #guitarset + on-the-fly torch.overdrive rendering
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
print("Is the Train Dataset IDMT? Answer :", config.train_and_valid_with_idmt)
print(f"Total Counts of training audio excerpts with fixed duration : {len(train_dataset)}")
print(f"Total Time_durtaion of training audio excerpts : {config.slice_length * len(train_dataset)}")




# S3. Choose loss function, SGD Optimizer, and validation metric. #Default = Multi-Scaled Spectral Loss(MSS) and Adam Optimizer
# Define training loss, and select FX types to be applied on wet input and wet recon audio.

if config.use_dist == True:
    if config.use_room_acoustic == True:
        signal_key = "audio_dist_room"
    else:
        signal_key = "audio_dist"   
else: #use_dist == False. But this case never happens in this script
    if  config.use_room_acoustic == True:
        signal_key = "audio_room"
    else : 
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

trainable_parameters_list = filter(lambda p: p.requires_grad, net.parameters())
num_trainable_parameters = sum([torch.numel(p) for p in trainable_parameters_list])
print("\n The num of trainable parameters in net : ", num_trainable_parameters)

# Main Training Loop (zero_grad, forward propagation with network, backpropagation, save parameters at ckp) #
time_start_train = time.time()
for epoch in tqdm(range(config.max_epoch), desc="epochs"):
    epoch += 1
    avg_loss_dry = 0
    avg_loss_wet = 0
    avg_loss_value = 0
    batches_in_one_epoch = int(len(train_dataset) / config.batch_size)
    print(f"There are {batches_in_one_epoch} batches per epoch, and {config.batch_size} audio excerpts per batch.")
    print(f"Start the training of {start_epoch + epoch}th epoch.")
    
    if config.distortion.use_pre_eq == False :
        print("Pre-Eq in Distortion Module in not being used. Only Power_Tanh works as Distortion")
    else:
        print("Distortion use both pre-eq and power-tanh series.")
        
    for batch_idx, batch in tqdm(enumerate(train_dataloader), desc="Batches") : # Iterations in one epoch
         
        # Clear the gradients
        # Feed Input audio to GPU (batch-wise)
        # GDDSP use only audio_wet_GT as input, and then reconstruct de-effected audio(=dry audio) by using Oscillators and Noise Generator.
        # GDDSP subsequently reconstructs effected audio(=wet audio) similar to input wet audio by rendering FX Chain on dry_recon.
        start_iter =time.time()
        print(f"Now processing {batch_idx}th/{len(train_dataloader)} batch in {start_epoch + epoch}th epoch.")
        print("Counts of 4_sec_sliced audio in one batch : ", batch["audio_wet_GT"].shape[0])
        batch["audio_dry_GT"] = batch["audio_dry_GT"].to(device) #This will be used to calculate loss between dry_GT and dry_recon
        batch["audio_wet_GT"] = batch["audio_wet_GT"].to(device) #This is input data, and will be used to calculate loss between wet_GT and wet_recon
        audio_recon = net(batch)
        loss_dry = loss(audio_recon["audio_dry"], batch["audio_dry_GT"]) # Loss at Intermediate Layer (only evaluate the quality of output of Oscillators and Noise Generator.) 
        loss_wet = loss(audio_recon[signal_key], batch["audio_wet_GT"]) # Loss at Final Layer (evaluate the quality of output of total net.)
        loss_value = loss_dry + loss_wet 
        # audio_recon : Reconstructed audio of net. It can be used for Self-Supervised Training of net.
        # latent : output of Decoder = input knob values of rule-based Synthesizers and rule-based FX-Chain.
        # batch_pitch_z_loud : output of Encoder = input of Decoder = Useful sonic charcters of input audio.
        optimizer.zero_grad()
        loss_value.backward()
        nn.utils.clip_grad_norm_(net.parameters(), max_norm)
        optimizer.step()  # Update model parameters
        avg_loss_dry += loss_dry.detach().item()
        avg_loss_wet += loss_wet.detach().item()
        avg_loss_value += loss_value.detach().item() # avg_loss = average loss of all audio data in one epoch
        print("Time Consuming for One Iteration is : ", time.time()-start_iter)
        
    scheduler.step()
    avg_loss_dry = avg_loss_dry/ len(train_dataloader) # Averaging on audios in batch was already done by MSS L1 Loss function
    avg_loss_wet = avg_loss_wet/ len(train_dataloader) # So to get loss per one audio, Just add process of Averaging loss on batches
    avg_loss_value = avg_loss_value / len(train_dataloader)
    wandb.log({"(joint) dry_loss VS epoch": avg_loss_dry,
               "(joint) wet_loss VS epoch": avg_loss_wet,
               "(joint) hybrid_loss VS epoch": avg_loss_value,
               "epoch" : start_epoch+epoch,
               "elapsed_time" : time.time() - time_start_train,
               "num_trainable_params" : num_trainable_parameters})
    
    
    ## save the parameters periodically
    if epoch % config.check_point_save_period == 0 and epoch > 100:
        if not os.path.exists(config.check_point_save_path + experiment_full_name):
            os.makedirs(config.check_point_save_path + experiment_full_name)
        ckp_path = config.check_point_save_path + experiment_full_name + f"/joint_loss_epoch{start_epoch + epoch}.pt"
        torch.save({
            'epoch': start_epoch + epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_dry' : avg_loss_dry,
            'loss_wet' : avg_loss_wet,
            'hybrid_loss': avg_loss_value
            }, ckp_path)
        print(f"The trained parameters are saved at {ckp_path}")
    
    ## save the sample audio reconstructed by GDDSP periodically with .wav format, and compare it with original audio of same data_idx.
    if epoch % config.sample_audio_recon_save_period == 0 and epoch > 100 :
        import soundfile as sf
        sr = config.sample_rate
        if not os.path.exists(config.sample_audio_recon_save_path + experiment_full_name):
            os.makedirs(config.sample_audio_recon_save_path + experiment_full_name)
        audio_path = config.sample_audio_recon_save_path + experiment_full_name + f"/useWH_epoch{start_epoch + epoch}"
        sample_original_audio_dry = batch["audio_dry_GT"][0].detach().cpu().numpy()
        sample_original_audio_dist = batch["audio_wet_GT"][0].detach().cpu().numpy()
        sf.write(audio_path + "_original_dry.wav", sample_original_audio_dry, sr)
        sf.write(audio_path + "_original_dist.wav", sample_original_audio_dist, sr)
        # batch["~"][0] = The first audio excerpt of final minibatch of present epoch.
        # selected sample audio excerpt differs by epoch. (as the train_dataloader use "shuffle")
        if config.use_room_acoustic == True:
            sample_recon_audio_dry_before_WH = audio_recon["audio_dry"][0].detach().cpu().numpy()
            sample_recon_audio_dist_after_WH_before_RE = audio_recon["audio_dist"][0].detach().cpu().numpy()
            sample_recon_audio_dist_after_WH_after_RE = audio_recon["audio_dist_room"][0].detach().cpu().numpy()
            sf.write(audio_path + "_recon_dry_before_WH+RE.wav", sample_recon_audio_dry_before_WH, sr)
            sf.write(audio_path + "_recon_dist_with_WH+RE_dereverb.wav", sample_recon_audio_dist_after_WH_before_RE, sr)
            sf.write(audio_path + "_recon_dist_with_WH+RE.wav", sample_recon_audio_dist_after_WH_before_RE, sr)
        else: #Reverb layer wasnt constructed
            sample_recon_audio_dry_before_WH = audio_recon["audio_dry"][0].detach().cpu().numpy()
            sample_recon_audio_dist_after_WH = audio_recon["audio_dist"][0].detach().cpu().numpy()
            sf.write(audio_path + "_recon_dry_before_WH.wav", sample_recon_audio_dry_before_WH, sr)
            sf.write(audio_path + "_recon_dist_with_WH.wav", sample_recon_audio_dist_after_WH, sr)
            
        print(f"The sample audio reconstructed by GDDSP is saved at {audio_path}")
    
    ## Calculate valid loss Periodically, and Trace Valid loss with wandb
    if epoch % config.valid_period  == 0 :
        avg_valid_loss_dry = 0
        avg_valid_loss_wet = 0
        avg_valid_loss_value = 0
        optimizer.zero_grad()  # Clear the gradients
        print(f"validation at {start_epoch + epoch}th epoch started.")
        with torch.no_grad():
            for batch_idx, valid_batch in tqdm(enumerate(valid_dataloader), desc="Batches"):
                print(f"{batch_idx+1}th valid batch is under processing. There are {len(valid_dataloader)} batches in valid dataset")
                print(f"There are {len(valid_dataset) // len(valid_dataloader)} audio excerpts in one minibatch of valid dataset")
                valid_batch["audio_dry_GT"] = valid_batch["audio_dry_GT"].to(device)
                valid_batch["audio_wet_GT"] = valid_batch["audio_wet_GT"].to(device)
                audio_recon = net(valid_batch)
                valid_loss_dry = loss(audio_recon["audio_dry"], valid_batch["audio_dry_GT"]) # Loss at Intermediate Layer (only evaluate the quality of output of Oscillators and Noise Generator.) 
                valid_loss_wet = loss(audio_recon[signal_key], valid_batch["audio_wet_GT"]) # Loss at Final Layer (evaluate the quality of output of total net.)
                valid_loss_value = valid_loss_dry + valid_loss_wet
                
                avg_valid_loss_dry += valid_loss_dry.detach().item()
                avg_valid_loss_wet += valid_loss_wet.detach().item()
                avg_valid_loss_value += valid_loss_value.detach().item()
        avg_valid_loss_dry = avg_valid_loss_dry / len(valid_dataloader)
        avg_valid_loss_wet = avg_valid_loss_wet / len(valid_dataloader)
        avg_valid_loss_value = avg_valid_loss_value / len(valid_dataloader)
        wandb.log({"(joint) valid_dry_loss VS epoch": avg_valid_loss_dry,
                   "(joint) valid_wet_loss VS epoch": avg_valid_loss_wet,
                   "(joint) valid_hybrid_loss VS epoch": avg_valid_loss_value,
                   "epoch" : start_epoch+epoch})
        print(f"validation at {start_epoch + epoch}th epoch finished.")
