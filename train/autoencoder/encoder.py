# Polyphonic Pitch Encoder ,Loudness Encoder, Z Encoder for GDDSP

### Encoder module 1 (pitch) ###
# This module is not my own structure. It is the Pre-Trained Lightweihgt Inst-Agnoistc Polyphonic Pitch Estimator (https://github.com/spotify/basic-pitch)
# In : audio signal
# Out : Frame-wise polyphonic pitches

### Encoder module 2 (loudness) ###
# This module is not a trainable neural net. It is just a rule-based Loudness Extractor. (rule = A-Weighted Integration of Power Spectrum)
# In : audio signal
# Out : Frame-wise Loudness 
# Output is not a note-wise feature.
# ex> If there are 3 different pitches in target frame7, then this Encoder will extract the total intensity of superposition of those notes)

### Encoder module 3 (z) ###
# In : audio signal
# Out : Some residual information related to timbre
# This Encoder does not separately encode the information of dry timbre and the information of FX settings.
# It just encode the "Overall" timbre.
# However in decoder parts there are three separated modules, two for reconstructing dry signal and one for inferring FX parameters.
#  (two : decoders for harmonic Osccilator and Noise,   one : decoder for FX layers)
#  Each decoder will learn by itself to focus on specific part of z-emb which is useful for each one's function.
#  i.e. z is not itself disentangled into dry/wet, but the mapping system(=z to Amplitude_harmonic, z to phase_harmonic, z to FX_parameters,...) will disentangle it.

import time
import scipy
import torch
import torchaudio
import tensorflow as tf
import torch
import torch.nn as nn
import numpy as np
import tensorflow as tf
import sys
sys.path.append("..")
from components.basic_pitch.inference_not_use_path import predict 
# This inferecne class does not need the directory of .wav file. It directly refer to the sliced audio Tensor and then return the pitch-seq existing in that Tensor.
from components.basic_pitch import ICASSP_2022_MODEL_PATH


class Encoder_pitch_loud_z(nn.Module):
    # input : config(in init) and input audio(in forward method)
    # output of forward : six f0-sequences, loud_sequences, z_sequences (frame-wise)
    # Temporal Resolution of frames < Temporal Resolution of Sample Points
    '''
    GDDSP writter's default config 
    frame_resolution = num of frame per second / num of sampling points per second  = 0.004
    it is, for calculation of loudness and z, use 250 sample points 
    
    '''
    def __init__(self, config):
        super().__init__()

        self.config = config
        if self.config.use_dist or self.config.use_room_acoustic or self.config.recon_dist_without_fx_chain:
            self.input_audio_key = "audio_wet_GT" #If you wanna target to recon Dist Audio, Regradless of Using W_H_Distortio Layer, input_audio_key="audio_wet_GT"
        else:
            self.input_audio_key = "audio_dry_GT"
        
        self.hop_length = int(config.sample_rate * config.frame_resolution)
        self.fx_hop_length = int(config.sample_rate * config.fx_frame_resolution)
        # frame_resolution = num of frame per second / num of sampling points per second
        # same hop_length must be fed to z_enc and loudness_enc, to unify the shape of ouput of encoders.
        self.loudness_extractor = LoudnessExtractor(
            sr=config.sample_rate, frame_length=self.hop_length, config=self.config
        )
        # frame_length(int) : Temporal Length of each window to calculate FFT or MFCC
        # n_fft : Temporal Length of each window to calculate FFT or MFCC, including Overlapping Region with Neighboring window.
        # (n_fft/2 +1) is same as total numbers of freq_bins in one fft.

        if config.use_z:
            # z : Timbre Information for Dry Sound Reconstruction System (System = Decoder which produce knobs for Oscillator and Noise Generator)
            self.encoder_z = Encoder_z(
                sample_rate=config.sample_rate,
                n_fft=config.n_fft,
                hop_length=self.hop_length,
                n_mels=config.n_mels,
                n_mfcc=config.n_mfcc,
                gru_units=config.gru_units,
                z_units=config.z_units,
                bidirectional=config.bidirectional,
                config = self.config
            )
        #if self.config.use_dist or self.config.use_room_acoustic : 
        # #room_acoustic uses internal instance vector of reverb, so dose not need encoded zfx varying by input audio
        if self.config.use_dist or self.config.use_room_acoustic :
        # zfx : Timbre Information for FX_Knob Inferring Net in Decoder = Timbre information used for MLP_Dense_layers for distortion, MLP_Dense_layers for delay. 
            self.encoder_zfx = Encoder_z(
                sample_rate=config.sample_rate,
                n_fft=config.fx_n_fft,
                hop_length=self.fx_hop_length,
                n_mels=config.fx_n_mels,
                n_mfcc=config.fx_n_mfcc,
                gru_units=config.gru_units,
                z_units=config.zfx_units,
                bidirectional=config.bidirectional,
                config = self.config   
            )
        if self.config.enc_pitch.structure == "light_basic_pitch" :
            self.encoder_pitch = tf.saved_model.load(str(ICASSP_2022_MODEL_PATH))
            
    
    def Encoding_poly_pitch_light(self, audio_input_batch): # Get the Existence Prob of 0~263th bin. (= A0~ C8 =  27.5Hz ~ 4186Hz)
        #### Pitch Encoding Function####
        # For pitch Encoding, Use Pre-trained Polyphonic Basic-Pitch model's Predict function.
        # Predict function of Basic-pitch model has been modified to afford "Batch-Wise Inference", "Direct reference to audio tensor from dataloader, not path of audio.wav"
        # Check Modified Predict function in ../train/components/basic_pitch/inference_not_use_path.py
        audio_input_batch = audio_input_batch.cpu().numpy() # no for loop. Batch-wise Pitch encoding
        pitch_prob_batch = predict(audio_input_batch, self.encoder_pitch)
        return pitch_prob_batch
    
    def forward(self, batch):
        #### Main  method of GDDSP_Encoder #### 
        #### Get Pitch_emb, Loud_emb, Z_emb, ZFX_emb, then add them into batch dictionary, then deliver it to decoder. ####
        start_enc = time.time()
        if isinstance(batch, dict) == False:
            batch = {self.input_audio_key : batch}
        
        #Batch-wise Input is dictionary of "audio_wet_GT", "audio_dry_GT" "gain"
        if self.config.enc_pitch.structure == "light_basic_pitch" :
            batch["pitch_prob"] = self.Encoding_poly_pitch_light(batch[self.input_audio_key])
        if self.config.use_z:
            batch["z"] = self.encoder_z(batch[self.input_audio_key])
        if self.config.use_dist or self.config.use_room_acoustic :
            batch["zfx"] = self.encoder_zfx(batch[self.input_audio_key])
        batch["loudness"] = self.loudness_extractor(batch[self.input_audio_key])
        
            
        #### Pitch_emb, Loud_emb, Z_emb, ZFX_emb has been produced ####

        ### Unify the Number of Frames of pitch, loud, z ### 
        if batch["pitch_prob"][0].shape[0] == batch["loudness"][0].shape[0] and batch["pitch_prob"][0].shape[0] == batch["z"][0].shape[0] :
            with torch.no_grad():
                batch["pitch_prob"] = torch.Tensor(batch["pitch_prob"]).to(batch[self.input_audio_key].device)
        else :
            # Upsample and apply the linear interpolation on pitch-emb so that its length of rows will be same as the ones of loud,z
            common_frame_nums = batch["loudness"][0].shape[0]
            original_pitch_frame_axis = np.arange(batch["pitch_prob"][0].shape[0])
            common_frame_axis =  np.linspace(0, batch["pitch_prob"][0].shape[0]-1, num = common_frame_nums)
            # original_pitch_frame_axis : start_Frame=0, end_frame=343, and those 344 frames are splitted inoto 344 points
            # common_frame_axis : start_frame =0, end_frame=343, but those 344frames_intervals are now splitted into 400 points
            interp_function = scipy.interpolate.interp1d(original_pitch_frame_axis, batch["pitch_prob"], axis=1) #axis = 1 : 2nd axis = frame axis. 
            # DO NOT Interpolate across freq bin axis(axis=-1)
            pitch_batch_interpolated = interp_function(common_frame_axis)
            with torch.no_grad():
                batch["pitch_prob"] = torch.Tensor(pitch_batch_interpolated).to(batch[self.input_audio_key].device)
        ### "Unifying the Time-axis length of loudness, pitch, z"  has been finsished ###
        
        ### Picking six reliable freq bins, local f0 weight sum, and Thresholding. (as the max degree of polyphony of each frame is 6 in guitar recording) ###
        if self.config.enc_pitch.threshold_top_six : # Among 264 one-sixth tone(6분음) per one time window, remain the six toppest Existence prob of one-sixth tone bin. 
            topk_num_per_frame = 6
            topk_prob_values , freq_bin_idx_with_top_prob = torch.topk(batch["pitch_prob"], topk_num_per_frame, dim=-1, largest=True, sorted=True) #pick 6 bins with highest probs(confidences) per frame per audio.
            # freq_bin_idx_with_top_prob : top_k indices of prob(confidence)s of pitches = top_k bin indices
            # dim = -1 = last axis = probs of freq bins
            freq_bin_idx_with_top_prob, sort_permutation_idx = torch.sort(freq_bin_idx_with_top_prob, dim=-1, descending=True, stable=False)

            if self.config.enc_pitch.weight_sum_for_top_six : ### If you want, Get the Weighted Center around each toppest f0 bin. ###
                cent_bank = torch.arange(264) * 100/3
                cent_bank = cent_bank.expand(batch["pitch_prob"].shape[0], batch["pitch_prob"].shape[1], 264).to(batch["pitch_prob"].device)
                lowest_bin_f0 = 27.5
                # cent bank w/ shape [Batch, Frames, 264] # 264 = 88 semitones splitted into 3 intervals per semitone
                # cent bank[:, :, i] = (i-1) * 33.3 cent = relative cent of i^th freq bin comparing to lowest f0 of basic-pitch encoder.
                
                weighted_cents = batch["pitch_prob"] * cent_bank #element-wise multiplication
                sum_radius = 4
                
                weighted_cents = nn.functional.pad(weighted_cents, (sum_radius , sum_radius)) #pad on freq_bin axis, to unfold by unit of neighboring 9 bins
                weighted_cents_grouped_around_each_bin = weighted_cents.unfold(-1, 2*sum_radius+1, 1) #w/ shape [Batch, Frames, 264, 9]
                weighted_cents_sum_around_each_bin = torch.sum(weighted_cents_grouped_around_each_bin, dim=-1) #w/ shape [Batch, Frames, 264]
                weighted_cents_sum_around_top_bin = torch.gather(weighted_cents_sum_around_each_bin, -1, freq_bin_idx_with_top_prob)  #w/ shape [Batch, Frames, 6]
                #torch_gather(A, -1 ,B) = along -1th axis of A, collect elements with 6 top confidences. B offers bin_indices to be collected
                
                padded_pitch_prob = nn.functional.pad(batch["pitch_prob"], (sum_radius , sum_radius)) #pad on freq_bin axis, to unfold by unit of neighboring 9 bins
                pitch_prob_grouped_around_each_bin = padded_pitch_prob.unfold(-1, 2*sum_radius+1, 1) #w/ shape [Batch, Frames, 264, 9]
                pitch_prob_sum_around_each_bin = torch.sum(pitch_prob_grouped_around_each_bin, dim=-1) #w/ shape [Batch, Frames, 264]
                pitch_prob_sum_around_top_bin = torch.gather(pitch_prob_sum_around_each_bin, -1, freq_bin_idx_with_top_prob)  #w/ shape [Batch, Frames, 6]

                normalized_weighted_cents_sum_around_top_bin = weighted_cents_sum_around_top_bin / pitch_prob_sum_around_top_bin
                weighted_f0_around_top_bin = lowest_bin_f0 * torch.pow(2 , normalized_weighted_cents_sum_around_top_bin/1200)

                # Thresholding Weight Sum of f0 (If Definite Prob Sum around top bin is low, then ignore that top bin(hz) even though it has relatively high prob.)
                weighted_f0_around_top_bin_thresholded = torch.where(pitch_prob_sum_around_top_bin > self.config.enc_pitch.confidence_thres,  weighted_f0_around_top_bin, torch.tensor(0.0, dtype=torch.float32, device=batch["pitch_prob"].device) )
                
                # Sort again to naturally link nth frame's highest note - n+1th frame's highest note 
                # and link nth frame's second highest note - n+1th frame's second highest note .... same for lowest
                # and link nth frame's 0.0hz to n+1th frame's 0.0hz
                # By Sorting Again aqfter thres, Move 0Hz (= constant phase = Cant hear = Silence) to lowest columns in six_f0
                weighted_f0_around_top_bin_thresholded, sort_permute_idx_after_thresholding  = torch.sort(weighted_f0_around_top_bin_thresholded, dim=-1, descending=True, stable=False)
                batch["six_f0"] = weighted_f0_around_top_bin_thresholded.to(batch[self.input_audio_key].device)
                
                if self.config.enc_pitch.confidence_to_decoder == True :
                    pitch_prob_sum_around_top_bin = torch.gather(pitch_prob_sum_around_top_bin, -1, sort_permute_idx_after_thresholding) #sort confidence again, to locate "conf of pitch under thres" into lower rows
                    batch["six_conf"] = pitch_prob_sum_around_top_bin.to(batch[self.input_audio_key].device)
            
            else: #If weight_sum config == False, JUST confirm six pitches with toppest confidences as true pitches. Dont calculate Sigma(cent*confidence)/Sigma(confidence)
                cent_bank = torch.arange(264) * 100/3
                cent_bank = cent_bank.expand(batch["pitch_prob"].shape[0], batch["pitch_prob"].shape[1], 264).to(batch["pitch_prob"].device)
                lowest_bin_f0 = 27.5
                cent_top_bins_per_frame = torch.gather(cent_bank, -1, freq_bin_idx_with_top_prob)
                f0_top_bins_per_frame_no_weight_sum = lowest_bin_f0  * torch.pow(2, cent_top_bins_per_frame/1200)
                batch["six_f0"] = f0_top_bins_per_frame_no_weight_sum.to(batch[self.input_audio_key].device)
        else:
            pass #We always pick & Threshold six pitches per frame, for Guitar Sound Reconstrunction
        elapsed_enc = time.time() - start_enc
        print("Time Consuming for Encoding is : ", elapsed_enc)
        return batch
    
    def produce_zfx(self, batch):
        zfx = self.encoder_zfx(batch[self.input_audio_key])
        return zfx

class Encoder_pitch_loud_z_mono(nn.Module):
    # input : config(in init) and input audio(in forward method)
    # output of forward : one f0-sequence w/ shape [batch, n_frames, 1], loud_sequence w/ shape [batch, n_frames, 1], z_sequence w/ shape [batch, n_frames, config.z_unit]
    # Num of Frames depend on frame_resolution and sample_rate in configuration.yaml
    # If frame_resoltuion = 0.004 -> one frame represents the time-length 0.004 -> one frame has (sr*0.004)points
    # Temporal Resolution of frames < Temporal Resolution of Sample Points
    '''
    GDDSP writter's default config 
    frame_resolution = num of frame per second / num of sampling points per second  = 0.004
    it is, for calculation of loudness and z, use 250 sample points 
    '''
    def __init__(self, config):
        super().__init__()
        self.config = config
        if self.config.use_dist or self.config.use_room_acoustic or self.config.recon_dist_without_fx_chain :
            self.input_audio_key = "audio_wet_GT" #If you wanna target to recon Dist Audio, Regradless of Using W_H_Distortio Layer, input_audio_key="audio_wet_GT"
        else:
            self.input_audio_key = "audio_dry_GT"
        self.hop_length = int(config.sample_rate * config.frame_resolution)
        self.fx_hop_length = int(config.sample_rate * config.fx_frame_resolution)
        # frame_resolution = num of frame per second / num of sampling points per second
        # same hop_length must be fed to z_enc and loudness_enc, to unify the shape of ouput of encoders.
        self.loudness_extractor = LoudnessExtractor(
            sr=config.sample_rate, frame_length=self.hop_length, config=self.config
        )
        if config.use_z:
            # z : Timbre Information for Dry Sound Reconstruction System (System = Decoder which produce knobs for Oscillator and Noise Generator)
            self.encoder_z = Encoder_z(
                sample_rate=config.sample_rate,
                n_fft=config.n_fft,
                hop_length=self.hop_length,
                n_mels=config.n_mels,
                n_mfcc=config.n_mfcc,
                gru_units=config.gru_units,
                z_units=config.z_units,
                bidirectional=config.bidirectional,
                config = self.config
            )
        if self.config.use_dist:
        # zfx : Timbre Information for FX_Knob Inferring Net in Decoder = Timbre information used for MLP_Dense_layers for distortion 
            self.encoder_zfx = Encoder_z(
                sample_rate=config.sample_rate,
                n_fft=config.fx_n_fft,
                hop_length=self.fx_hop_length,
                n_mels=config.fx_n_mels,
                n_mfcc=config.fx_n_mfcc,
                gru_units=config.gru_units,
                z_units=config.zfx_units,
                bidirectional=config.bidirectional,
                config = self.config   
            )
        if self.config.enc_pitch.structure == "light_basic_pitch" :
            self.encoder_pitch = tf.saved_model.load(str(ICASSP_2022_MODEL_PATH))
            
    def Encoding_poly_pitch_light(self, audio_input_batch): # Get the Existence Prob of 0~263th bin. (= A0~ C8 =  27.5Hz ~ 4186Hz. 88-semitones keys)
        audio_input_batch = audio_input_batch.cpu().numpy() # no for loop. Batch-wise Pitch encoding
        pitch_prob_batch = predict(audio_input_batch, self.encoder_pitch)
        return pitch_prob_batch
    
    def forward(self, batch):
        #### Main  method of GDDSP_Encoder #### 
        #### Get Pitch_emb, Loud_emb, Z_emb, ZFX_emb, then add them into batch dictionary, then deliver it to decoder. ####
        start_enc = time.time()
        if isinstance(batch, dict): #Batch-wise Input is dictionary of "audio_wet_GT", "audio_dry_GT" "gain"
            if self.config.enc_pitch.structure == "light_basic_pitch" :
                batch["pitch_prob"] = self.Encoding_poly_pitch_light(batch[self.input_audio_key])
            if self.config.use_z:
                batch["z"] = self.encoder_z(batch[self.input_audio_key])
            if self.config.use_dist:
                batch["zfx"] = self.encoder_zfx(batch[self.input_audio_key])
            batch["loudness"] = self.loudness_extractor(batch[self.input_audio_key])
        else:
            batch_dict = {}
            if self.config.enc_pitch.structure == "light_basic_pitch" : # When we try to get model structure and n_params with torch_summary, use this option with simple tensor input [B, sr*4]
                batch_dict["pitch_prob"] = self.Encoding_poly_pitch_light(batch)
            if self.config.use_z:
                batch_dict["z"] = self.encoder_z(batch)
            if self.config.use_dist:
                batch_dict["zfx"] = self.encoder_zfx(batch)
            batch_dict["loudness"] = self.loudness_extractor(batch)
            batch = batch_dict
                
        batch["loudness"] = self.loudness_extractor(batch[self.input_audio_key])
        #### Pitch_emb, Loud_emb, Z_emb, ZFX_emb has been produced ####

        ### Unify the Number of Frames of pitch, loud, z ###
        
        if batch["pitch_prob"].shape[1] == batch["loudness"].shape[1] and batch["pitch_prob"].shape[1] == batch["z"].shape[1] :
            # shape[0]:Batch Axis, shape[1]:n_frame axis.
            with torch.no_grad():
                batch["pitch_prob"] = torch.Tensor(batch["pitch_prob"]).to(batch[self.input_audio_key].device)
        else :
            # Upsample and apply the linear interpolation on pitch-emb so that its length of rows will be same as the ones of loud,z
            common_frame_nums = batch["loudness"].shape[1]
            original_pitch_frame_axis = np.arange(batch["pitch_prob"][0].shape[0])
            common_frame_axis =  np.linspace(0, batch["pitch_prob"][0].shape[0]-1, num = common_frame_nums) # Split 0~pitch_nframe real-number range with discrete steps of loundess_emb. (to interpolate)
            interp_function = scipy.interpolate.interp1d(original_pitch_frame_axis, batch["pitch_prob"], axis=1) # axis1 = frame axis. 
            # DO NOT Interpolate across freq bin axis(axis = -1)
            pitch_batch_interpolated = interp_function(common_frame_axis)
            with torch.no_grad(): # Pitch Encoding system is already pre-trained. Dont need grad backprop
                batch["pitch_prob"] = torch.Tensor(pitch_batch_interpolated).to(batch[self.input_audio_key].device)
        ### "Unifying the Time-axis length of loudness, pitch, z"  has been finsished ###
        
        ### Picking One Argmax Freq Bin, And Calculate logal weight Sum of Prob * Pitch ###
        topk_num_per_frame = 1 # Pick 1 pitch per frame, For Monophonic Audio and Mono-DDSP
        
        topk_prob_values , freq_bin_idx_with_top_prob = torch.topk(batch["pitch_prob"], topk_num_per_frame, dim=-1, largest=True, sorted=True) #pick 1 bin with highest probs(confidences) per frame per audio.
        cent_bank = torch.arange(264) * 100/3
        cent_bank = cent_bank.expand(batch["pitch_prob"].shape[0], batch["pitch_prob"].shape[1], 264).to(batch["pitch_prob"].device)
        lowest_bin_f0 = 27.5
        # cent bank w/ shape [Batch, Frames, 264] # 264 = 88 semitones splitted into 3 intervals per semitone
        # cent bank[:, :, i] = (i-1) * 33.3 cent = relative cent of i^th freq bin comparing to lowest f0 of basic-pitch encoder.
        
        weighted_cents = batch["pitch_prob"] * cent_bank #element-wise multiplication
        sum_radius = 4
        weighted_cents = nn.functional.pad(weighted_cents, (sum_radius , sum_radius)) #pad on freq_bin axis, to unfold by unit of neighboring 9 bins
        weighted_cents_grouped_around_each_bin = weighted_cents.unfold(-1, 2*sum_radius+1, 1) #w/ shape [Batch, Frames, 264, 9]
        weighted_cents_sum_around_each_bin = torch.sum(weighted_cents_grouped_around_each_bin, dim=-1) #w/ shape [Batch, Frames, 264]
        weighted_cents_sum_around_top_bin = torch.gather(weighted_cents_sum_around_each_bin, -1, freq_bin_idx_with_top_prob)  #w/ shape [Batch, Frames, 6]
        # Even Though we only need weight sum around One top bin, calculate all sums around all bins first.
        # It's because once you leave only one tio bin in prob tensor and cent tensor, than cannot load the information of neighboring bins.
        #torch_gather(A, -1 ,B) = along -1th axis of A, collect elements with 6 top confidences. B offers bin_indices to be collected
        
        padded_pitch_prob = nn.functional.pad(batch["pitch_prob"], (sum_radius , sum_radius)) #pad on freq_bin axis, to unfold by unit of neighboring 9 bins
        pitch_prob_grouped_around_each_bin = padded_pitch_prob.unfold(-1, 2*sum_radius+1, 1) #w/ shape [Batch, Frames, 264, 9]
        pitch_prob_sum_around_each_bin = torch.sum(pitch_prob_grouped_around_each_bin, dim=-1) #w/ shape [Batch, Frames, 264]
        pitch_prob_sum_around_top_bin = torch.gather(pitch_prob_sum_around_each_bin, -1, freq_bin_idx_with_top_prob)  #w/ shape [Batch, Frames, 6]
        normalized_weighted_cents_sum_around_top_bin = weighted_cents_sum_around_top_bin / pitch_prob_sum_around_top_bin
        weighted_f0_around_top_bin = lowest_bin_f0 * torch.pow(2 , normalized_weighted_cents_sum_around_top_bin/1200)

        # Thresholding Weight Sum of f0 (If Definite Prob Sum around top bin is low, then ignore that top bin(hz) even though it has relatively high prob.)
        weighted_f0_around_top_bin_thresholded = torch.where(pitch_prob_sum_around_top_bin > self.config.enc_pitch.confidence_thres,  weighted_f0_around_top_bin, torch.tensor(0.0, dtype=torch.float32, device=batch["pitch_prob"].device) )
        batch["one_f0"] = weighted_f0_around_top_bin_thresholded.to(batch[self.input_audio_key].device)
        
        if self.config.enc_pitch.confidence_to_decoder == True :
            batch["one_conf"] = pitch_prob_sum_around_top_bin.to(batch[self.input_audio_key].device)
        else:
            pass
        elapsed_enc = time.time() - start_enc
        print("Time Consuming for Encoding is : ", elapsed_enc)
        return batch
    def produce_zfx(self, batch):
        zfx = self.encoder_zfx(batch[self.input_audio_key])
        return zfx


#### Loudness ####
class LoudnessExtractor(nn.Module):
    # This code was copied from DDSP_implemented_with_pytorch (https://github.com/sweetcocoa/ddsp-pytorch)
    # For Detail, See https://github.com/sweetcocoa/ddsp-pytorch/blob/master/components/loudness_extractor.py
    # license of this code : © 2020 Jongho Choi (sweetcocoa@snu.ac.kr, BS Student @ Seoul National Univ.), Sungho Lee (dlfqhsdugod1106@gmail.com, BS Student @ Postech.)
    # license of Original DDSP structure :  © 2019 Google LLC.
    def __init__(self,
                 sr = 22050,
                 frame_length = 64,
                 attenuate_gain = 2.,
                 config = None):
        
        super(LoudnessExtractor, self).__init__()
        self.config = config
        if self.config.use_dist or self.config.use_room_acoustic or self.config.recon_dist_without_fx_chain :
            self.input_audio_key = "audio_wet_GT"
        else:
            self.input_audio_key = "audio_dry_GT"
        self.sr = sr
        self.frame_length = frame_length
        self.n_fft = self.frame_length * 5
        # frame_length(int) : Temporal Length of each window to calculate FFT or MFCC
        # n_fft : Temporal Length of each window to calculate FFT or MFCC, including Overlapping Region with Neighboring window.
        # n_fft is also same as total numbers of freq_bins in one fft(window)
        # frame_length and n_fft must be consistently set with same value in enc_z, enc_loud, and pre_eq of distortion, 
        # , to unify shape of latents produced in various intermediate layers in GDDSP.
        
        self.attenuate_gain = attenuate_gain
        self.smoothing_window = nn.Parameter(torch.hann_window(self.n_fft, dtype = torch.float32), requires_grad = False)

    

    def torch_A_weighting(self, FREQUENCIES, min_db = -45.0):
        """
        Compute A-weighting weights in Decibel scale (codes from librosa) and 
        transform into amplitude domain (with DB-SPL equation).
        
        Argument: 
            FREQUENCIES : tensor of frequencies to return amplitude weight
            min_db : mininum decibel weight. appropriate min_db value is important, as 
                exp/log calculation might raise numeric error with float32 type. 
        
        Returns:
            weights : tensor of amplitude attenuation weights corresponding to the FREQUENCIES tensor.
        """
        # Calculate A-weighting in Decibel scale.
        FREQUENCY_SQUARED = FREQUENCIES ** 2 
        const = torch.tensor([12200, 20.6, 107.7, 737.9]) ** 2.0
        WEIGHTS_IN_DB = 2.0 + 20.0 * (torch.log10(const[0]) + 4 * torch.log10(FREQUENCIES)
                               - torch.log10(FREQUENCY_SQUARED + const[0])
                               - torch.log10(FREQUENCY_SQUARED + const[1])
                               - 0.5 * torch.log10(FREQUENCY_SQUARED + const[2])
                               - 0.5 * torch.log10(FREQUENCY_SQUARED + const[3]))
        # Set minimum Decibel weight.
        if min_db is not None:
            WEIGHTS_IN_DB = torch.max(WEIGHTS_IN_DB, torch.tensor([min_db], dtype = torch.float32))
        # Transform Decibel scale weight to amplitude scale weight.
        weights = torch.exp(torch.log(torch.tensor([10.], dtype = torch.float32)) * WEIGHTS_IN_DB / 10) 
        
        return weights

        
    def forward(self, batch):
        """
        Compute A-weighted Loudness Extraction
        Input:
            z['audio'] : batch of time-domain signals
        Output:
            output_signal : Weighted Sum of Loudess of freq bins, per frame.
        """
        if isinstance(batch, dict):
            input_signal = batch[self.input_audio_key]
        else:
            input_signal = batch
        paded_input_signal = nn.functional.pad(input_signal, (self.frame_length * 2, self.frame_length * 2))
        # (self.frame_length * 2, self.frame_length * 2) = (pad_left, pad_right)
        # This function only pads on last dimension(time dimenstion of audio. not the batch_idx dimension.)
        sliced_signal = paded_input_signal.unfold(1, self.n_fft, self.frame_length)
        sliced_windowed_signal = sliced_signal * self.smoothing_window
        
        # sliced_signal : tensor w/ shape : (batch, windows, sample_points per window)
        # slieced_windowed_signal : same as sliced_signal, but the edge points are attenuated by Hann Window.
        
        SLICED_SIGNAL = torch.fft.fft(sliced_windowed_signal, dim=-1)
        # The Original implementor of torch_based_DDSP used torch.rfft(sliced_windowed_signal, n=1, onesided=false)
        # But the torch.rfft function has been removed since pytorch 16.0
        # for now(2023.09),  
        # torch.rfft(x, onesided=false) is equivalent to torch.fft.fft(x)
        # torch.rfft(x, onesided=true(=default)) is equivalent to torch.fft.rfft(x)
        SLICED_SIGNAL_LOUDNESS_SPECTRUM = torch.zeros(SLICED_SIGNAL.shape).to(input_signal.device)
        SLICED_SIGNAL_LOUDNESS_SPECTRUM = SLICED_SIGNAL[:, :, :].real**2 + SLICED_SIGNAL[:, :, :].imag**2
        # SLICED_SIGNAL_LOUDNESS_SPECTRUM : Squared Magnitude of SLICED_SINGAL. Axis [:, :, :] menans [batch, frame_idx, freq bins in that frame]
     
        freq_bin_size = self.sr / self.n_fft #freq interval(hz) betweein neighboring fft bin
        FREQUENCIES = torch.tensor([(freq_bin_size * i) % (0.5 * self.sr) for i in range(self.n_fft)])
        # freq_bin_size * i = center freq value of i^th bin
        # % 0.5*self.sr -> Cut the freq value over (self.sr/2), to handle Aliasing (Read Nyquist Theorem)
        
        A_WEIGHTS = self.torch_A_weighting(FREQUENCIES).to(input_signal.device)
        A_WEIGHTED_SLICED_SIGNAL_LOUDNESS_SPECTRUM = SLICED_SIGNAL_LOUDNESS_SPECTRUM * A_WEIGHTS
        A_WEIGHTED_SLICED_SIGNAL_LOUDNESS = torch.sqrt(torch.sum(A_WEIGHTED_SLICED_SIGNAL_LOUDNESS_SPECTRUM, 2)) / self.n_fft * self.attenuate_gain
        return A_WEIGHTED_SLICED_SIGNAL_LOUDNESS




#### Z and ZFX Encoder Structure (Structured Shared, but arguments n_fft, n_mel, n_mfcc might be different)####
class Encoder_z(nn.Module):
    # This code was copied from DDSP_implemented_with_pytorch (https://github.com/sweetcocoa/ddsp-pytorch)
    # For Detail, See https://github.com/sweetcocoa/ddsp-pytorch/blob/master/train/network/autoencoder/encoder.py
    # license of this code : © 2020 Jongho Choi (sweetcocoa@snu.ac.kr, BS Student @ Seoul National Univ.), Sungho Lee (dlfqhsdugod1106@gmail.com, BS Student @ Postech.)
    # license of Original DDSP structure :  © 2019 Google LLC.
    
    def __init__(
        self,
        n_fft,
        hop_length,
        sample_rate=22050,
        n_mels=128,
        n_mfcc=30,
        gru_units=512,
        z_units=16,
        bidirectional=False,
        config=None
    ):
        super().__init__()
        self.config = config
        if self.config.use_dist or self.config.use_room_acoustic or self.config.recon_dist_without_fx_chain :
            self.input_audio_key = "audio_wet_GT"
        else:
            self.input_audio_key = "audio_dry_GT"

        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            log_mels=True,
            melkwargs=dict(
                n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, f_min=20.0, f_max=8000.0,
            ),
        )

        self.norm = nn.InstanceNorm1d(n_mfcc, affine=True)
        self.permute = lambda x: x.permute(0, 2, 1)
        self.gru = nn.GRU(
            input_size=n_mfcc,
            hidden_size=gru_units,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.dense = nn.Linear(gru_units * 2 if bidirectional else gru_units,  z_units)

    def forward(self, batch):
        if isinstance(batch, dict):
            x = batch[self.input_audio_key]
        else:
            x = batch
        x = self.mfcc(x)
        x = x[:, :, :-1]
        x = self.norm(x)
        x = self.permute(x)
        x, _ = self.gru(x) # Trainable GRU # _ = hidden cell of GRU
        x = self.dense(x) # Trainable FC layer (In: GRU unit, Out:Z or ZFX Unit)
        return x
    
    
