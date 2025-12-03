import torch
import torch.nn as nn
import sys, os, glob
from autoencoder.decoder import Decoder_Oscillator_Noise_FX, Decoder_Oscillator_Noise_FX_mono
from autoencoder.encoder import Encoder_pitch_loud_z, Encoder_pitch_loud_z_mono
from components.harmonic_oscillator import HarmonicOscillator
from components.filtered_noise import FilteredNoise
from components.fx_chain import W_H_Distortion # Wienner-Hammerstein Structure representing Distortion Effect. (Linear EQ - NonLinear Tanh - Linear EQ)
from components.fx_chain import TrainableFIRReverb

class GDDSP_wet(nn.Module): 
    # GDDSP_dry : encoder + decoder + HarmoncOscillator + Noise Generator 
    # Input : audio tensor w/ shape: (batch, sr*duration)
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = Encoder_pitch_loud_z(config)
        self.decoder = Decoder_Oscillator_Noise_FX(config)\
        
        hop_length = int(config.sample_rate * config.frame_resolution)
        # Num_Points Advancement per Frame. (for loud, pitch, z processing, Not for zfx)
        # default hop = sr * frame_resol = 22050 * 0.004 = 88
        # One frame represents "frame_resolution seconds" and "hop_length points"
        self.harmonic_oscillator = HarmonicOscillator(
            sr=config.sample_rate, frame_length=hop_length
        )
        self.filtered_noise = FilteredNoise(frame_length=hop_length)
        if self.config.use_dist == True:
            if self.config.input_dist_amount_random == True:
                self.distortion = W_H_Distortion(config)
            else:
                self.distortion_shared_on_all_data = W_H_Distortion(config)
                
        if self.config.use_room_acoustic == True :
            self.room_acoustic_shared_on_all_data = TrainableFIRReverb(config)
    
    def handle_audio_length_inconsistency(self, audio_recon):
        input_GT_audio_length = self.config.sample_rate * self.config.slice_length
        if audio_recon.shape[-1] > input_GT_audio_length:
            audio_recon = audio_recon[..., : input_GT_audio_length] #cut audio_recon
        elif audio_recon.shape[-1] == input_GT_audio_length :
            pass
        else : # The case that audio_recon is shorter than original input audio. Pad at Last Axis(=Time Axis = Sample points Axis)
            lacking_length = input_GT_audio_length - audio_recon.shape[-1]
            audio_recon = torch.nn.functional.pad(audio_recon, (0,  lacking_length), mode='constant', value=0) #lengthen audio_recon (by padding at right side of time axis)
        return audio_recon
        
    
    def forward(self, batch):
        """
        batch : input of encoder (=sliced audio 1-D waveforms, and corresponding source_idx, slice_idx. these indices are related to idx of __getitem__ of dataloader)
            shape : dict{"audio":~, "source_idx":~, "slice_idx":~}
        batch["audio"] : sliced audio 1-D waveforms. (=Ground Truth Input audio) 
            shape : (batch, sr*sliced_length)
        batch_pitch_z_loud : output of encoder (= input of decoder)
            shape : dict{"audio" :~, "six_f0":~, "loud" :, "z" : ~} 
        latent : output of decoder = knob values fed into Oscillators, Noise Generator, and FX Chain.(FX Chain = Distortion - Room Acoustic in series)
            shape : dict{"six_f0: ~, "a" : ~, "c" :~ "H" : ~, "dist_pre_eq" : ~, "dist_ctanh" : ~, "room_acoustic" : ~}
            
            six_f0 : same as six_f0 in batch_pitch_z_loud (skip connection from encoder to Harmonic Oscillator)
            
            a : global amp of sinusoid = amp of fundametal wav of each string. 
            a : list[a1, a2, a3, a4, a5, a6]
            a1 : Sinusoid Amplitude sequence represents highest pitch of each frame. tensor w/ shape (batch, frames)
            
            c : relative amps of harmonics  w/ shape ( batch, frames, 6, config.n_harmonics ) 
            c : list[c1, c2, c3, c4, c5, c6]
            c1 : tensor w/ shape (batch, frames, config.n_harmonics)
            
            H : Frame-Wise Freq bin value of Noise Filter in Frequency-Time Domain (This will be convolved with Frame-Wise White Noise)
                shape : ( batch, frames, config.n_freq )
            
            dist_pre_eq : Frame-Wise coefficients of pre_eq transfer function included in distortion
                shape : (batch, frames, config.distortion.n_eq_band)
            dist_ctanh = Frame-Wise coefficients of power tanh series included in distortion(this tanh function give audio nonlinearity)
                shape  : (batch, frames, config.distortion.n_tanh)
        """
        batch_pitch_z_loud = self.encoder(batch) 
        latent = self.decoder(batch_pitch_z_loud)

        # There is no trainable parameter in harmonic_oscillator.
        # harmonic_oscillator works by built-in rule. (\SIGMA A_k * sin(phi_k))
        # So there is no need to make distinct oscillator classes or instances for construction of each harmonic waveform.
        harmonic_1st = self.harmonic_oscillator(latent, string_idx=1) # harmonic_1st is sum of fundamental and harmonics reepresenting highest pitch_seq.
        harmonic_2nd = self.harmonic_oscillator(latent, string_idx=2) # string_idx helps harmoni_oscillator to pick proper f0, a, and c.
        harmonic_3rd = self.harmonic_oscillator(latent, string_idx=3)
        harmonic_4th = self.harmonic_oscillator(latent, string_idx=4)
        harmonic_5th = self.harmonic_oscillator(latent, string_idx=5)
        harmonic_6th = self.harmonic_oscillator(latent, string_idx=6)
        
        # Final Reconstruction of Audio
        total_harmonic = harmonic_1st + harmonic_2nd + harmonic_3rd + harmonic_4th + harmonic_5th + harmonic_6th
        noise = self.filtered_noise(latent)
        # why do we use noise[:, : total_harmonic.shape[-1] ?
        # The temporal duration of Noise might be little bit longer than harmonics.
        # It's because the filter-convolution's tail in noise generation process has not been cut.
        audio_dry = total_harmonic + noise[:, : total_harmonic.shape[-1]] # audio_dry : Clean audio reconstructed by GDDSP 
        
        # Handle the length-inconsistency between reconstructed audio and input_audio
        audio_dry = self.handle_audio_length_inconsistency(audio_dry)
        # length-inconsistency can bring out error in pre-EQ of distortion
        
        audio_recon = dict(audio_dry = audio_dry)
        # role of dict-key "audio_dry" is same as Jongho's DDSP's "audio_synth" (Jongho's DDSP : https://github.com/sweetcocoa/ddsp-pytorch)
        
        # Reconstruction of guitar audio including effects
        if self.config.use_dist: # use distortion
            if self.config.input_dist_amount_random == True:
                audio_dist = self.distortion(audio_dry, latent_dict_fed_from_Decoder = latent)
                audio_dist = self.handle_audio_length_inconsistency(audio_dist)
                audio_recon["audio_dist"] = audio_dist
            else: # use One Internal Distortion Knobs to figure out Fixed Distortion shared on all data
                audio_dist = self.distortion_shared_on_all_data(audio_dry, latent_dict_fed_from_Decoder=None) 
                # Do not Receive Knob latent from Decoder which differs by audio. Instead, use common global ctanh and preEQ to all recon audio.
                audio_dist = self.handle_audio_length_inconsistency(audio_dist)
                audio_recon["audio_dist"] = audio_dist
            if self.config.use_room_acoustic:
                audio_dist_room = self.room_acoustic_shared_on_all_data(audio_dist)
                audio_dist_room = self.handle_audio_length_inconsistency(audio_dist_room)
                audio_recon["audio_dist_room"] = audio_dist_room
            else:
                pass
            
        elif self.config.use_room_acoustic: #do not use distortion
            audio_room = self.room_acoustic_shared_on_all_data(audio_dry)
            audio_room = self.handle_audio_length_inconsistency(audio_room)
            audio_recon["audio_room"] = audio_room
        else:
            pass
        return audio_recon
        # audio_recon = {audio_dry, audio_dist, audio_dist_room} or {audio_dry, audio_dist} or {audio_dry, audio_room}
        
    def produce_zfx(self, batch):
        # Extract the Intermediate Latent vector "zfx" from Input Distorted Audio
        # Then Check whetere that vector is useful for reverse-Estimation of Colour Knob Valud and Gain Knob Value of torchaudio.overdrive
        zfx = self.encoder.produce_zfx(batch)
        zfx_average_along_frames = torch.mean(zfx, dim=1) #dim=0 : batch dim.  dim=1: frames dim,  dim=2: z_unit dim(len=16)
        return zfx_average_along_frames

    def produce_distortion_knobs(self, batch):
        zfx = self.encoder.produce_zfx(batch)
        latent_zfx = self.decoder.mlp_zfx(zfx)
        latent_zfx_average_along_frames = torch.mean(latent_zfx, dim=1)
        dist_pre_eq  = self.decoder.dense_zfx_average_to_dist_pre_eq_knob(latent_zfx_average_along_frames)
        dist_pre_eq = GDDSP_wet.modified_sigmoid(dist_pre_eq)
        dist_ctanh = self.decoder.dense_zfx_average_to_dist_ctanh_knob(latent_zfx_average_along_frames)# One Scalar per audio. Can Have Range 0~5
        pre_amp_before_tanh =  5 * GDDSP_wet.modified_sigmoid(dist_ctanh)[..., -1].unsqueeze(1)
        dist_ctanh =  5 * GDDSP_wet.modified_sigmoid(dist_ctanh)[..., :-1]
        knobs = torch.cat((dist_pre_eq, pre_amp_before_tanh, dist_ctanh), dim=-1)
        return knobs
        
    @staticmethod
    def modified_sigmoid(a):
        a = a.sigmoid()
        a = a.pow(2.3026)  # log10
        a = a.mul(2.0)
        a.add_(1e-7)
        return a
class GDDSP_wet_mono(nn.Module): 
    # Monophonic Version of GDDSP
    # Enc produces One f0 seq per audio, Dec's MLP,GRU,Dense produces One a-seq and c-seq for One Oscillator
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = Encoder_pitch_loud_z_mono(config)
        self.decoder = Decoder_Oscillator_Noise_FX_mono(config)
        hop_length = int(config.sample_rate * config.frame_resolution)
        # Num_Points Advancement per Frame. (for loud, pitch, z processing, Not for zfx)
        # default hop = sr * frame_resol = 22050 * 0.004 = 88
        # One frame represents "frame_resolution seconds" and "hop_length points"
        self.harmonic_oscillator = HarmonicOscillator(
            sr=config.sample_rate, frame_length=hop_length
        )
        self.filtered_noise = FilteredNoise(frame_length=hop_length)
        if self.config.use_dist == True:
            if self.config.input_dist_amount_random == True:
                self.distortion = W_H_Distortion(config)
            else:
                self.distortion_shared_on_all_data = W_H_Distortion(config)       
        if self.config.use_room_acoustic == True :
            self.room_acoustic_shared_on_all_data = TrainableFIRReverb(config)
    def handle_audio_length_inconsistency(self, audio_recon):
        input_GT_audio_length = self.config.sample_rate * self.config.slice_length
        if audio_recon.shape[-1] > input_GT_audio_length:
            audio_recon = audio_recon[..., : input_GT_audio_length] #cut audio_recon
        elif audio_recon.shape[-1] == input_GT_audio_length :
            pass
        else : # The case that audio_recon is shorter than original input audio. Pad at Last Axis(=Time Axis = Sample points Axis)
            lacking_length = input_GT_audio_length - audio_recon.shape[-1]
            audio_recon = torch.nn.functional.pad(audio_recon, (0,  lacking_length), mode='constant', value=0) #lengthen audio_recon (by padding at right side of time axis)
        return audio_recon
        
    def forward(self, batch):
        batch_pitch_z_loud = self.encoder(batch) 
        latent = self.decoder(batch_pitch_z_loud)

        total_harmonic = self.harmonic_oscillator(latent, string_idx=None)
        # By Setting string_idx = None, oscillator can recognize that a, c latent given from decoder is not the list of [a_1st,..a_6th], [c_1st, .. c_6th] but is just single element a_1st and c_1st

        noise = self.filtered_noise(latent)
        audio_dry = total_harmonic + noise[:, : total_harmonic.shape[-1]] # audio_dry : Clean audio reconstructed by GDDSP 
        
        # Handle the length-inconsistency between reconstructed audio and input_audio
        audio_dry = self.handle_audio_length_inconsistency(audio_dry)
        # length-inconsistency can bring out error in pre-EQ of distortion
        
        audio_recon = dict(audio_dry = audio_dry)
        
        # Reconstruction of guitar audio including effects
        if self.config.use_dist: # use distortion
            if self.config.input_dist_amount_random == True:
                audio_dist = self.distortion(audio_dry, latent_dict_fed_from_Decoder = latent)
                audio_dist = self.handle_audio_length_inconsistency(audio_dist)
                audio_recon["audio_dist"] = audio_dist
            else: # use One Internal Distortion Knobs to figure out Fixed Distortion shared on all data
                audio_dist = self.distortion_shared_on_all_data(audio_dry, latent_dict_fed_from_Decoder=None) 
                # Do not Receive Knob latent from Decoder which differs by audio. Instead, use common global ctanh and preEQ to all recon audio.
                audio_dist = self.handle_audio_length_inconsistency(audio_dist)
                audio_recon["audio_dist"] = audio_dist
            if self.config.use_room_acoustic:
                audio_dist_room = self.room_acoustic_shared_on_all_data(audio_dist)
                audio_dist_room = self.handle_audio_length_inconsistency(audio_dist_room)
                audio_recon["audio_dist_room"] = audio_dist_room
            else:
                pass
        elif self.config.use_room_acoustic: #do not use distortion
            audio_room = self.room_acoustic_shared_on_all_data(audio_dry)
            audio_room = self.handle_audio_length_inconsistency(audio_room)
            audio_recon["audio_room"] = audio_room
        else:
            pass
        return audio_recon
        # audio_recon = {audio_dry, audio_dist, audio_dist_room} or {audio_dry, audio_dist} or {audio_dry, audio_room}
    def produce_zfx(self, batch):
        # Extract the Intermediate Latent vector "zfx" from Input Distorted Audio
        # Then Check whetere that vector is useful for reverse-Estimation of Colour Knob Valud and Gain Knob Value of torchaudio.overdrive
        zfx = self.encoder.produce_zfx(batch)
        zfx_average_along_frames = torch.mean(zfx, dim=1) #dim=0 : batch dim.  dim=1: frames dim,  dim=2: z_unit dim(len=16)
        return zfx_average_along_frames
    def produce_distortion_knobs(self, batch):
        zfx = self.encoder.produce_zfx(batch)
        latent_zfx = self.decoder.mlp_zfx(zfx)
        latent_zfx_average_along_frames = torch.mean(latent_zfx, dim=1)
        dist_pre_eq  = self.decoder.dense_zfx_average_to_dist_pre_eq_knob(latent_zfx_average_along_frames)
        dist_pre_eq = GDDSP_wet.modified_sigmoid(dist_pre_eq)
        dist_ctanh = self.decoder.dense_zfx_average_to_dist_ctanh_knob(latent_zfx_average_along_frames)# One Scalar per audio. Can Have Range 0~5
        pre_amp_before_tanh =  5 * GDDSP_wet.modified_sigmoid(dist_ctanh)[..., -1].unsqueeze(1)
        dist_ctanh =  5 * GDDSP_wet.modified_sigmoid(dist_ctanh)[..., :-1]
        knobs = torch.cat((dist_pre_eq, pre_amp_before_tanh, dist_ctanh), dim=-1)
        return knobs
    @staticmethod
    def modified_sigmoid(a):
        a = a.sigmoid()
        a = a.pow(2.3026)  # log10
        a = a.mul(2.0)
        a.add_(1e-7)
        return a
