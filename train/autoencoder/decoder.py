# This Script contains 3 types or decoder“

### decoder module 1 (harmonic) ###
# In : Pitch-seq, loudness-seq, timbre emb
# Out : Frame-wise Amplitudes and phases which will be fed into six harmonic osciilators

### decoder module 2 (noise) ###
# In : Pitch-seq, loudness-seq, timbre emb
# Out : Frame-wise Noise Amplitudes, Noise Filter parameters 


### decoder module 3 (FX) ###
# In : loudness-seq , timbre emb 
# Out : Distortion parameters, Delay parameters, Reverb parameters"


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MLP(nn.Module):
    """
    MLP (Multi-layer Perception). 
    One layer consists of what as below:
        - 1 Dense Layer
        - 1 Layer Norm
        - 1 ReLU (Activation)
    constructor arguments :
        n_input : dimension of input : dimension of encoded audio feature to be fed into MLP_Decoder. i.e. input of MLP has shape [Batch, n_input]
        n_units : dimension of hidden unit
        n_layer : depth of MLP (the number of layers)
        relu : relu (default : nn.ReLU, can be changed to nn.LeakyReLU, nn.PReLU for example.)
    input(x): torch.tensor w/ shape(B, ... , n_input)
    output(x): torch.tensor w/ (B, ..., n_units)
    """

    def __init__(self, n_input, n_units, n_layer, relu=nn.ReLU, inplace=True):
        super().__init__()
        self.n_layer = n_layer # Depth of MLP
        self.n_input = n_input # Input size of MLP = (Batch, Frames, n_input)
        self.n_units = n_units # output size of MLP = (Batch, Frames, n_units)
        self.inplace = inplace

        self.add_module(
            f"mlp_layer1",
            nn.Sequential(
                nn.Linear(n_input, n_units),
                nn.LayerNorm(normalized_shape=n_units),
                relu(inplace=self.inplace),
            ),
        )

        for i in range(2, n_layer + 1):
            self.add_module(
                f"mlp_layer{i}",
                nn.Sequential(
                    nn.Linear(n_units, n_units),
                    nn.LayerNorm(normalized_shape=n_units),
                    relu(inplace=self.inplace),
                ),
            )

    def forward(self, x):
        for i in range(1, self.n_layer + 1):
            x = self.__getattr__(f"mlp_layer{i}")(x)
        return x


class Decoder_Oscillator_Noise_FX(nn.Module):
    """
    Decoder.

    Constructor arguments: 
        use_z : (Bool), if True, Decoder will use z as input.
        mlp_units: 512
        mlp_layers: 3
        z_units: 16
        n_harmonics: 101
        n_fft : 441 (for noise generator)
        n_freq: 65 (for noise generator)
        gru_units: 512
        bidirectional: False

    input: a dict object which contains key-values below = output of encoder
        six_f0 : fundamental frequencies with top-six confidences, collected from polyphonic f0 encoder. torch.tensor w/ shape(B, time)
        z : (optional) residual information. torch.tensor w/ shape(B, time, z_units)
        loudness : torch.tensor w/ shape(B, time)

        # z_units represent various timbral characters of input audio 

    output : a dict object which contains key-values below
        six_f0 : same as input
        a : Global amplitude of Waveform of each string.(=amplitude of fundamental wave) torch.tensor w/ shape(B, frames) which satisfies a > 0
        c : Relative amplitude of 2nd, 3rd,... harmonics of each string. torch.tensor w/ shape(B, frames, n_harmonics) which satisfies sum(c) == 1
             ex> a*c1 = amplitude of fundamental, a*c2 = amplitude of second-harmonic 
        H : Noise filter coefficient in frequency domain. torch.tensor w/ shape(B, frames, filter_coeff_length)
        H will be convolved with FFT(white-noise sequence), to reconstruct generalized aperiodic component of input audio.
        dist_pre_eq : adjustable knob parameters(freq bin coefficients) for pre eq included in distortion module.
        dist_ctanh : adjustable coefficients of power tanh series included in distortion module(default : produce twenty ctanh)
        delay : adjustable knob parameters for delay module.
        reverb : adjustable knob parameters for reverb module
    
        Decoder must deliver six_f0_seq to Harmonic_oscillator without any modifying. (= Skip-Connection of Pitch Information)
    """

    def __init__(self, config):
        super().__init__()

        self.config = config
        ### Decoding Flow ###
        # MLP -> GRU -> DENSE (In: z, loud, six f0  -> Out: a, c, H) 
        # or MLP -> AVERAGE -> DENSE (In: zfx -> Out : dist_pre_eq_bins, dist_ctanh)
        # For Dry_Synthesizing, (six_f0, loud, z) -> fed into eight MLPs -> fed into GRU -> fed into Dense(FC) to produce a, c , H
        # For FX_Estimation, (zfx)->fed into MLP -> Averaged on frames to get Global FX latent -> fed into Dense(FC) to produce ctanh, pre_dist_Eq
        self.mlp_f0_1st = MLP(n_input=1, n_units=config.mlp_units, n_layer=config.mlp_layers) #produce the latent vector of the highest sequence of f0. this latent will be fed into oscillator which represents the high E string in the guitar.
        self.mlp_f0_2nd = MLP(n_input=1, n_units=config.mlp_units, n_layer=config.mlp_layers) #output of mlp w/ shape : (Batch, frame, config.mlp_units)
        self.mlp_f0_3rd = MLP(n_input=1, n_units=config.mlp_units, n_layer=config.mlp_layers)
        self.mlp_f0_4th = MLP(n_input=1, n_units=config.mlp_units, n_layer=config.mlp_layers)
        self.mlp_f0_5th = MLP(n_input=1, n_units=config.mlp_units, n_layer=config.mlp_layers)
        self.mlp_f0_6th = MLP(n_input=1, n_units=config.mlp_units, n_layer=config.mlp_layers)
        self.mlp_loudness = MLP(n_input=1, n_units=config.mlp_units, n_layer=config.mlp_layers)
        
        if self.config.enc_pitch.confidence_to_decoder == True:
            self.mlp_conf_1st = MLP(n_input=1, n_units=config.mlp_units, n_layer=config.mlp_layers)
            self.mlp_conf_2nd = MLP(n_input=1, n_units=config.mlp_units, n_layer=config.mlp_layers)
            self.mlp_conf_3rd = MLP(n_input=1, n_units=config.mlp_units, n_layer=config.mlp_layers)
            self.mlp_conf_4th = MLP(n_input=1, n_units=config.mlp_units, n_layer=config.mlp_layers)
            self.mlp_conf_5th = MLP(n_input=1, n_units=config.mlp_units, n_layer=config.mlp_layers)
            self.mlp_conf_6th = MLP(n_input=1, n_units=config.mlp_units, n_layer=config.mlp_layers)
            if config.use_z:
                self.mlp_z = MLP(
                    n_input=config.z_units, n_units=config.mlp_units, n_layer=config.mlp_layers
                )
                self.num_mlp_before_GRU = 14
                # Declare input dimension of GRU which produces activiation fed into Dense_harmonic and Dense_Noise
                # GRU Layer for Dry Synthesizing Receives 14 concatenated mlp_output (= latent_f0_1st~6th, latent_confi_1st~6th, z_latent and loud_latent)
            else:
                self.num_mlp_before_GRU = 13
        else:
            if config.use_z:
                self.mlp_z = MLP(
                    n_input=config.z_units, n_units=config.mlp_units, n_layer=config.mlp_layers
                )
                self.num_mlp_before_GRU = 8
                # Declare input dimension of GRU which produces activiation fed into Dense_harmonic and Dense_Noise
                # GRU Layer for Dry Synthesizing Receives 8 concatenated mlp_output (= latent_f0_1st~6th, z_latent and loud_latent)
            else:
                self.num_mlp_before_GRU = 7
            
        # Declare MLP_zfx which is the first decoding layer to process zfx
        if self.config.use_dist or self.config.use_room_acoustic :
            if self.config.zfx_mlp_unit_is_half_of_other_mlps :
                self.mlp_zfx = MLP(
                    n_input=config.zfx_units, n_units=config.mlp_units//2, n_layer=config.mlp_layers//2)
            else:
                self.mlp_zfx = MLP(
                    n_input=config.zfx_units, n_units=config.mlp_units, n_layer=config.mlp_layers)

        # Delcare GRU to handle time-dependence of pitch,lound,timbre of Music
        self.gru_dry = nn.GRU(
            input_size= self.num_mlp_before_GRU  * config.mlp_units, # = output dim of MLPs processing f0s,loud,z = input dim of GRU
            hidden_size=config.gru_units,
            num_layers=1,
            batch_first=True,
            bidirectional=config.bidirectional,
        )
        self.mlp_gru = MLP(
            n_input=config.gru_units * 2 if config.bidirectional else config.gru_units,
            n_units=config.mlp_units,
            n_layer=config.mlp_layers,
            inplace=True,
        )
        
        # Declare Dense(FC) layer to produce Interpretable Semantic Knobs of Osicllaotors ,Noise Generators , and W_H_Distortion
        # one element for overall loudness of one harmonic oscillator
        self.dense_harmonic_1st = nn.Linear(config.mlp_units, config.n_harmonics + 1)
        self.dense_harmonic_2nd = nn.Linear(config.mlp_units, config.n_harmonics + 1)
        self.dense_harmonic_3rd = nn.Linear(config.mlp_units, config.n_harmonics + 1)
        self.dense_harmonic_4th = nn.Linear(config.mlp_units, config.n_harmonics + 1)
        self.dense_harmonic_5th = nn.Linear(config.mlp_units, config.n_harmonics + 1)
        self.dense_harmonic_6th = nn.Linear(config.mlp_units, config.n_harmonics + 1)
        # dense_harmonic : final layer to produce amplitudes of fundamental wav and harmonics for each string. (max : 6 strings)

        self.dense_filter_H =  nn.Linear(config.mlp_units, config.n_freq)
        # dense_filter_H : final layer to produce freq bin coefficients of Noise Generator.
        
        if self.config.use_dist and config.input_dist_amount_random : 
            # If dist_amount_random==False -> Fit One Distortion-Set to all audio in dataset 
            # -> Dont need to decode knobs. Just use one internal knob-set in W_H_distortion
            if config.distortion.time_varying == False: # Simply feed latent_z to global_distortion_producing_layers
                if self.config.zfx_mlp_unit_is_half_of_other_mlps :
                    self.dense_zfx_average_to_dist_pre_eq_knob = nn.Linear(config.mlp_units//2, config.distortion.n_eq_band)
                    self.dense_zfx_average_to_dist_ctanh_knob = nn.Linear(config.mlp_units//2, config.distortion.n_tanh + 1) # Additional 1 unit = pre-amp amount
                else:
                    self.dense_zfx_average_to_dist_pre_eq_knob = nn.Linear(config.mlp_units, config.distortion.n_eq_band)
                    self.dense_zfx_average_to_dist_ctanh_knob = nn.Linear(config.mlp_units, config.distortion.n_tanh + 1) # Addi
            else:
                # The case that distortion amount can change by temporal frame, as well as change by audio.
                # Let distortion_producing_layers to refer to output of GRU(in: z, loud, f0), to catch the time-varying character of distortion
                if self.config.zfx_mlp_unit_is_half_of_other_mlps :
                    self.dense_zfx_framewise_to_dist_pre_eq_knob  = nn.Linear(config.mlp_units//2, config.distortion.n_eq_band) #produce coefficients of freq bins of pre-EQ of W_H_Distortion
                    self.dense_zfx_framewise_to_dist_ctanh_knob  = nn.Linear(config.mlp_units//2, config.distortion.n_tanh + 1) # Additional 1 unit = pre-amp amount
                else:
                    self.dense_zfx_framewise_to_dist_pre_eq_knob  = nn.Linear(config.mlp_units, config.distortion.n_eq_band)
                    self.dense_zfx_framesiwe_to_dist_ctanh_knob  = nn.Linear(config.mlp_units, config.distortion.n_tanh + 1)     
                # Input of two layers above : GRU(MLP(z), MLP(1st f0 ~ 6th f0), MLP(loudness))
        

    def forward(self, batch):
        import time
        start_dec = time.time()
        
        #batch = dictionary including outputs of encoder and original audio-excerpt
        #batch["six_f0"].shape = (batch_size, timesteps, 6)
        f0_1st = batch["six_f0"][...,0].unsqueeze(-1) # 0 th pitch is the highest pitch of each time-frame
        f0_2nd = batch["six_f0"][...,1].unsqueeze(-1)
        f0_3rd = batch["six_f0"][...,2].unsqueeze(-1)
        f0_4th = batch["six_f0"][...,3].unsqueeze(-1)
        f0_5th = batch["six_f0"][...,4].unsqueeze(-1)
        f0_6th = batch["six_f0"][...,5].unsqueeze(-1)
        loudness = batch["loudness"].unsqueeze(-1)

        latent_f0_1st = self.mlp_f0_1st(f0_1st) # latent_~~~ = output of MLP = output of First Layer of Decoder
        latent_f0_2nd = self.mlp_f0_2nd(f0_2nd)
        latent_f0_3rd = self.mlp_f0_3rd(f0_3rd)
        latent_f0_4th = self.mlp_f0_4th(f0_4th)
        latent_f0_5th = self.mlp_f0_5th(f0_5th)
        latent_f0_6th = self.mlp_f0_6th(f0_6th)
        latent_loudness = self.mlp_loudness(loudness)
        
        if self.config.enc_pitch.confidence_to_decoder == True:
            conf_1st = batch["six_conf"][...,0].unsqueeze(-1)
            conf_2nd = batch["six_conf"][...,1].unsqueeze(-1)
            conf_3rd = batch["six_conf"][...,2].unsqueeze(-1)
            conf_4th = batch["six_conf"][...,3].unsqueeze(-1)
            conf_5th = batch["six_conf"][...,4].unsqueeze(-1)
            conf_6th = batch["six_conf"][...,5].unsqueeze(-1)
            latent_conf_1st = self.mlp_conf_1st(conf_1st)
            latent_conf_2nd = self.mlp_conf_2nd(conf_2nd)
            latent_conf_3rd = self.mlp_conf_3rd(conf_3rd)
            latent_conf_4th = self.mlp_conf_4th(conf_4th)
            latent_conf_5th = self.mlp_conf_5th(conf_5th)
            latent_conf_6th = self.mlp_conf_6th(conf_6th)
            if self.config.use_z:
                z = batch["z"]
                latent_z = self.mlp_z(z)
                latent_total_dry = torch.cat((latent_f0_1st, latent_f0_2nd, latent_f0_3rd, latent_f0_4th, latent_f0_5th, latent_f0_6th, \
                    latent_conf_1st, latent_conf_2nd, latent_conf_3rd, latent_conf_4th, latent_conf_5th, latent_conf_6th, latent_z, latent_loudness), dim=-1)
            else:
                latent_total_dry = torch.cat((latent_f0_1st, latent_f0_2nd, latent_f0_3rd, latent_f0_4th, latent_f0_5th, latent_f0_6th, \
                    latent_conf_1st, latent_conf_2nd, latent_conf_3rd, latent_conf_4th, latent_conf_5th, latent_conf_6th, latent_loudness), dim=-1)
        else:  
            if self.config.use_z:
                z = batch["z"]
                latent_z = self.mlp_z(z)
                latent_total_dry = torch.cat((latent_f0_1st, latent_f0_2nd, latent_f0_3rd, latent_f0_4th, latent_f0_5th, latent_f0_6th, latent_z, latent_loudness), dim=-1)
            else:
                latent_total_dry = torch.cat((latent_f0_1st, latent_f0_2nd, latent_f0_3rd, latent_f0_4th, latent_f0_5th, latent_f0_6th, latent_loudness), dim=-1)

        ## feed the output of various MLPs into One Shared GRU ##
        gru_applied_latent_total_dry, (h) = self.gru_dry(latent_total_dry) # h : hidden state of nn.GRU, undergoing gate function
        gru_applied_latent_total_dry = self.mlp_gru(gru_applied_latent_total_dry)
        # Now the latent is expected to consider "time dependence" between frames in one audio.
        
        ## With Dense(FC), produce semantically interpretable knobs for Harmonic Oscillators ##
        amplitude_1st = self.dense_harmonic_1st(gru_applied_latent_total_dry)
        amplitude_2nd = self.dense_harmonic_2nd(gru_applied_latent_total_dry)
        amplitude_3rd = self.dense_harmonic_3rd(gru_applied_latent_total_dry)
        amplitude_4th = self.dense_harmonic_4th(gru_applied_latent_total_dry)
        amplitude_5th = self.dense_harmonic_5th(gru_applied_latent_total_dry)
        amplitude_6th = self.dense_harmonic_5th(gru_applied_latent_total_dry)
        #[.., 0] pick the amplitude of fundamental wave, from the array of [amp_fund, amp_harm_2nd, amp_harm_3rd ,..]
        # a = Global Amplitude of waveform = Amplitude of fundamental = Amplitude of 0th harmonic= amplitde[batch, time, 0]
        a_1st = amplitude_1st[..., 0] 
        a_1st = Decoder_Oscillator_Noise_FX.modified_sigmoid(a_1st)
        a_2nd = amplitude_2nd[..., 0]
        a_2nd = Decoder_Oscillator_Noise_FX.modified_sigmoid(a_2nd)
        a_3rd = amplitude_3rd[..., 0]
        a_3rd = Decoder_Oscillator_Noise_FX.modified_sigmoid(a_3rd)
        a_4th = amplitude_4th[..., 0]
        a_4th = Decoder_Oscillator_Noise_FX.modified_sigmoid(a_4th)
        a_5th = amplitude_5th[..., 0]
        a_5th = Decoder_Oscillator_Noise_FX.modified_sigmoid(a_5th)
        a_6th = amplitude_6th[..., 0]
        a_6th = Decoder_Oscillator_Noise_FX.modified_sigmoid(a_6th)
        # c = relative amplitudes of harmonics (num of harmonics per waveform = written in config)
        # Use F.Softmax, to ensure that the sum of relative amplitdues of all harmonics = 1
        c_1st = F.softmax(amplitude_1st[..., 1:], dim=-1) 
        c_2nd = F.softmax(amplitude_2nd[..., 1:], dim=-1)
        c_3rd = F.softmax(amplitude_3rd[..., 1:], dim=-1)
        c_4th = F.softmax(amplitude_4th[..., 1:], dim=-1)
        c_5th = F.softmax(amplitude_5th[..., 1:], dim=-1)
        c_6th = F.softmax(amplitude_6th[..., 1:], dim=-1)
        c_1st = c_1st.permute(0, 2, 1)  # to match the shape of harmonic oscillator's input argument.
        c_2nd = c_2nd.permute(0, 2, 1)
        c_3rd = c_3rd.permute(0, 2, 1)
        c_4th = c_4th.permute(0, 2, 1) 
        c_5th = c_5th.permute(0, 2, 1) 
        c_6th = c_6th.permute(0, 2, 1) 
        a = [a_1st, a_2nd, a_3rd, a_4th, a_5th, a_6th]
        c = [c_1st, c_2nd, c_3rd, c_4th, c_5th, c_6th]
        ## With Dense(FC), produce semantically interpretable knobs for Harmonic Oscillators (FINISHED) ##
        
        ## With Dense(FC), produce semantically interpretable knobs for Noise Generator ##
        # H : Coefficients Frequency Transfer Function (= EQ) whose num of bins is decided by config.n_freq
        H = self.dense_filter_H(gru_applied_latent_total_dry) 
        H = Decoder_Oscillator_Noise_FX.modified_sigmoid(H)
        
        ## Produce semantically interpretable knobs for W_H Distortion ##
        if self.config.use_dist and self.config.input_dist_amount_random:
            zfx = batch["zfx"]
            latent_zfx = self.mlp_zfx(zfx)
            latent_zfx_average_along_frames = torch.mean(latent_zfx, dim=1)  # Get global mlp(zfx) that ignores difference between frames in one audio.

            if self.config.distortion.time_varying == False: #dist amount can vary by audio, but not vary by time_frame.
                dist_pre_eq  = self.dense_zfx_average_to_dist_pre_eq_knob (latent_zfx_average_along_frames)
                dist_pre_eq = Decoder_Oscillator_Noise_FX.modified_sigmoid(dist_pre_eq)
                dist_ctanh = self.dense_zfx_average_to_dist_ctanh_knob (latent_zfx_average_along_frames)# One Scalar per audio. Can Have Range 0~5
                if self.config.distortion.ctanh_sigmoid == True: #Sigmoid ensures ctanh to be positive
                    pre_amp_before_tanh =  5 * Decoder_Oscillator_Noise_FX.modified_sigmoid(dist_ctanh)[..., -1]
                    dist_ctanh =  5 * Decoder_Oscillator_Noise_FX.modified_sigmoid(dist_ctanh)[..., :-1]
                    # dist_ctanh = 5 * sigmoid(dist_ctanh) -> Limit Range of dist_ctanh -inf ~ +inf into positive number 0~5
                    # If I limit range of coeffcient of each tanh term 0~1, It is hard to realize Post-Amplifying effect of Distortion
            else:
                # Produce Time-Varying adujustable knob values for W_H_distortion Function. (Get FX Knob varying by temporal frames) #
                dist_pre_eq = self.dense_zfx_framewise_to_dist_pre_eq_knob (gru_applied_latent_total_dry)
                dist_pre_eq = Decoder_Oscillator_Noise_FX.modified_sigmoid(dist_pre_eq)
                dist_ctanh = self.dense_zfx_framewise_to_dist_ctanh_knob (gru_applied_latent_total_dry)
                if self.config.distortion.ctanh_sigmoid == True: #Sigmoid ensures ctanh to be positive
                    pre_amp_before_tanh =  5 * Decoder_Oscillator_Noise_FX.modified_sigmoid(dist_ctanh)[..., -1]
                    dist_ctanh =  5 * Decoder_Oscillator_Noise_FX.modified_sigmoid(dist_ctanh)[... , :-1]
                    # dist_ctanh = 5 * sigmoid(dist_ctanh) -> Limit Range of dist_ctanh -inf ~ +inf into positive number 0~5
                    # If I limit range of coeffcient of each tanh term 0~1, It is hard to realize Amplifying effect of Distortion
            print("Time Consuming for Decoding is : ", time.time()-start_dec)
            return dict(six_f0=batch["six_f0"], a=a, c=c, H=H, dist_pre_eq = dist_pre_eq, dist_ctanh = dist_ctanh, pre_amp_before_tanh = pre_amp_before_tanh)
        else: 
            print("Time Consuming for Decoding is : ", time.time()-start_dec)
            return dict(six_f0=batch["six_f0"], a=a, c=c, H=H) # There is no Dense Layer to estimate Distortion Knobrs, neither W_H_Distortion Layers.
        # dict(six_f0=batch["six_f0"], a=a, c=c, H=H) : latent that is fed into Six Oscillators and the Noise generator
        # dict(dist_pre_eq = dist_pre_eq, dist_ctanh = dist_ctanh) : latent that is fed into W_H_Distortion which infer Non-linear Effect applied on Input Audio
        # Feed the raw f0 into the Oscillator with a skip-connection, following the same procedure as in the original DDSP.
    @staticmethod
    def modified_sigmoid(a):
        a = a.sigmoid()
        a = a.pow(2.3026)  # log10
        a = a.mul(2.0)
        a.add_(1e-7)
        return a

class Decoder_Oscillator_Noise_FX_mono(nn.Module):
    """
    Monophonic Decoder which produce variables to be fed into Harmonic Oscillator, Noise Generator, and W-H Distortion

    Constructor arguments: 
        use_z : (Bool), if True, Decoder will use z as input.
        mlp_units: 512
        mlp_layers: 3
        z_units: 16
        n_harmonics: 101 /121 / 141 (if 101 -> produce 1 amplitude-seq for fund wav and produce 100 amplitude-seqs for harmonics)
        n_fft : 441 (for noise generator)
        n_freq: 40 (for noise generator)
        gru_units: 512
        bidirectional: False
    input: a dict object which contains key-values below = output of encoder
        one_f0 : One fundamental frequency per frame with top confidence. torch.tensor w/ shape(B, time)
        z : (optional) residual information. torch.tensor w/ shape(B, time, z_units)
        loudness : torch.tensor w/ shape(B, time)
        # z_units represent various timbral characters of input audio 

    output : a dict object which contains key-values below
        one_f0 : same as input (Delivered with skip-connection)
        a : Global amplitude of Waveform of each string.(=amplitude of fundamental wave) torch.tensor w/ shape(B, frames) which satisfies a > 0
        c : Relative amplitude of 2nd, 3rd,... harmonics of each string. torch.tensor w/ shape(B, frames, n_harmonics) which satisfies sum(c) == 1
             ex> a*c1 = amplitude of fundamental, a*c2 = amplitude of second-harmonic 
        H : Noise filter coefficient in frequency domain. torch.tensor w/ shape(B, frames, filter_coeff_length)
        H will be convolved with FFT(white-noise sequence), to reconstruct generalized aperiodic component of input audio.
        dist_pre_eq : adjustable knob parameters(freq bin coefficients) for pre eq included in distortion module.
        dist_pre_amp : adjustable volume multiplication coefficient before tanh formula
        dist_ctanh : adjustable coefficients of power tanh series included in distortion module(default : produce twenty ctanh)
        Decoder must deliver six_f0_seq to Harmonic_oscillator without any modifying. (= Skip-Connection of Pitch Information)
    """

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.mlp_f0_1st = MLP(n_input=1, n_units=config.mlp_units, n_layer=config.mlp_layers) #In : f0 per frame. w/ shape [B, frames, 1] , Out : latent information related to f0 w/ shape [B, frames, mlp_unit]
        self.mlp_loudness = MLP(n_input=1, n_units=config.mlp_units, n_layer=config.mlp_layers)
        
        if self.config.enc_pitch.confidence_to_decoder == True:
            self.mlp_conf_1st = MLP(n_input=1, n_units=config.mlp_units, n_layer=config.mlp_layers)
            if config.use_z:
                self.mlp_z = MLP(
                    n_input=config.z_units, n_units=config.mlp_units, n_layer=config.mlp_layers
                )
                self.num_mlp_before_GRU = 4
                # GRU Layer for Dry Synthesizing Receives 4 concatenated mlp_output (= latent_f0, latent_confidence, latent_z, latent_loud)
            else:
                self.num_mlp_before_GRU = 3
        else:
            if config.use_z:
                self.mlp_z = MLP(
                    n_input=config.z_units, n_units=config.mlp_units, n_layer=config.mlp_layers
                )
                self.num_mlp_before_GRU = 3
                # GRU Layer for Dry Synthesizing Receives 3 concatenated mlp_output (= latent_f0, latent_z, latent_loud)
            else:
                self.num_mlp_before_GRU = 2
                
        # Declare MLP_zfx which is the first decoding layer to process zfx
        if self.config.use_dist:
            if self.config.zfx_mlp_unit_is_half_of_other_mlps :
                self.mlp_zfx = MLP(
                    n_input=config.zfx_units, n_units=config.mlp_units//2, n_layer=config.mlp_layers//2)
            else:
                self.mlp_zfx = MLP(
                    n_input=config.zfx_units, n_units=config.mlp_units, n_layer=config.mlp_layers)
        # Delcare GRU to handle time-dependence of pitch,lound,timbre of Music
        self.gru_dry = nn.GRU(
            input_size= self.num_mlp_before_GRU  * config.mlp_units, # = output dim of MLPs processing f0s,loud,z = input dim of GRU
            hidden_size=config.gru_units,
            num_layers=1,
            batch_first=True,
            bidirectional=config.bidirectional,
        )
        self.mlp_gru = MLP(
            n_input=config.gru_units * 2 if config.bidirectional else config.gru_units,
            n_units=config.mlp_units,
            n_layer=config.mlp_layers,
            inplace=True,
        )
        
        # Declare Dense(FC) layer to produce Interpretable Semantic Knobs of Osicllaotors ,Noise Generators , and W_H_Distortion
        # one element for overall loudness of one harmonic oscillator
        self.dense_harmonic_1st = nn.Linear(config.mlp_units, config.n_harmonics + 1)
        # dense_harmonic : final layer to produce amplitude("a") of fundamental wav and amplitudes("c") of harmonics for one monophonic string

        self.dense_filter_H =  nn.Linear(config.mlp_units, config.n_freq)
        # dense_filter_H : final layer to produce freq bin coefficients("H") fed into Noise Generator.
        
        if self.config.use_dist and config.input_dist_amount_random : 
            # If dist_amount_random==False -> Fit One Distortion-Set to all audio in dataset 
            # -> Dont need to decode knobs. Just use one internal knob-set in W_H_distortion
            if config.distortion.time_varying == False: # Simply feed latent_z to global_distortion_producing_layers
                if self.config.zfx_mlp_unit_is_half_of_other_mlps :
                    self.dense_zfx_average_to_dist_pre_eq_knob = nn.Linear(config.mlp_units//2, config.distortion.n_eq_band)
                    self.dense_zfx_average_to_dist_ctanh_knob = nn.Linear(config.mlp_units//2, config.distortion.n_tanh + 1) # Additional 1 unit = pre-amp amount
                else:
                    self.dense_zfx_average_to_dist_pre_eq_knob = nn.Linear(config.mlp_units, config.distortion.n_eq_band)
                    self.dense_zfx_average_to_dist_ctanh_knob = nn.Linear(config.mlp_units, config.distortion.n_tanh + 1) # Addi
            else:
                # Let distortion_producing_layers to refer to output of GRU(in: z, loud, f0), to catch the time-varying character of distortion
                if self.config.zfx_mlp_unit_is_half_of_other_mlps :
                    self.dense_zfx_framewise_to_dist_pre_eq_knob = nn.Linear(config.mlp_units//2, config.distortion.n_eq_band) #produce coefficients of freq bins of pre-EQ of W_H_Distortion
                    self.dense_zfx_framewise_to_dist_ctanh_knob = nn.Linear(config.mlp_units//2, config.distortion.n_tanh + 1) # Additional 1 unit = pre-amp amount
                else:
                    self.dense_zfx_framewise_to_dist_pre_eq_knob = nn.Linear(config.mlp_units, config.distortion.n_eq_band)
                    self.dense_zfx_framesiwe_to_dist_ctanh_knob = nn.Linear(config.mlp_units, config.distortion.n_tanh + 1)
                    
    def forward(self, batch):
        import time
        start_dec = time.time()
        f0_1st = batch["one_f0"] # Monophonic Decoder receive one freq seq per one audio in batch. batch["one_f0"].shape = [B, frames, 1] (Do not Need Unsqueeze)
        loudness = batch["loudness"].unsqueeze(-1) # By unsqueezing, shape [B,frames] -> [B, frames, 1]. to feed 1 element per frame into MLP_f0

        latent_f0_1st = self.mlp_f0_1st(f0_1st) # latent_~~~ = output of MLP = output of First Layer of Decoder
        latent_loudness = self.mlp_loudness(loudness)
        
        if self.config.enc_pitch.confidence_to_decoder == True:
            conf_1st = batch["one_conf"] # Do not need Unsqueeze
            latent_conf_1st = self.mlp_conf_1st(conf_1st)
            if self.config.use_z:
                z = batch["z"]
                latent_z = self.mlp_z(z)
                latent_total_dry = torch.cat((latent_f0_1st, latent_conf_1st,  latent_z, latent_loudness), dim=-1)
            else:
                latent_total_dry = torch.cat((latent_f0_1st, latent_conf_1st, latent_loudness), dim=-1)
        else:  
            if self.config.use_z:
                z = batch["z"]
                latent_z = self.mlp_z(z)
                latent_total_dry = torch.cat((latent_f0_1st,  latent_z, latent_loudness), dim=-1)
            else:
                latent_total_dry = torch.cat((latent_f0_1st,  latent_loudness), dim=-1)

        ## feed the output of various MLPs into One Shared GRU ##
        gru_applied_latent_total_dry, (h) = self.gru_dry(latent_total_dry) # h : hidden state of nn.GRU, undergoing gate function
        gru_applied_latent_total_dry = self.mlp_gru(gru_applied_latent_total_dry)
        # Now the latent is expected to consider "time dependence" between frames in one audio.
        
        ## With Dense(FC), produce semantically interpretable knobs for Harmonic Oscillators ##
        amplitude_1st = self.dense_harmonic_1st(gru_applied_latent_total_dry)
        #[.., 0] pick the amplitude of fundamental wave, from the array of [amp_fund, amp_harm_2nd, amp_harm_3rd ,..]
        # a = Global Amplitude of waveform = Amplitude of fundamental = Amplitude of 0th harmonic= amplitde[batch, time, 0]
        a_1st = amplitude_1st[..., 0] 
        a_1st = Decoder_Oscillator_Noise_FX.modified_sigmoid(a_1st)
        # c = relative amplitudes of harmonics (num of harmonics per waveform = written in config)
        # Use F.Softmax, to ensure that the sum of relative amplitdues of all harmonics = 1
        c_1st = F.softmax(amplitude_1st[..., 1:], dim=-1) # indexing 1: -> exclude index0(amp of fundamental wav)
        c_1st = c_1st.permute(0, 2, 1)  # to match the shape of harmonic oscillator's input argument.
        ## With Dense(FC), produce semantically interpretable knobs for Harmonic Oscillators (FINISHED) ##
        
        ## With Dense(FC), produce semantically interpretable knobs for Noise Generator ##
        # H : Coefficients Frequency Transfer Function (= EQ) whose num of bins is decided by config.n_freq
        H = self.dense_filter_H(gru_applied_latent_total_dry) 
        H = Decoder_Oscillator_Noise_FX.modified_sigmoid(H)
        
        ## Produce semantically interpretable knobs for W_H Distortion ##
        if self.config.use_dist and self.config.input_dist_amount_random :
            zfx = batch["zfx"]
            latent_zfx = self.mlp_zfx(zfx)
            latent_zfx_average_along_frames = torch.mean(latent_zfx, dim=1)  # Get global mlp(zfx) that ignores difference between frames in one audio.
            if self.config.distortion.time_varying == False:
                dist_pre_eq  = self.dense_zfx_average_to_dist_pre_eq_knob (latent_zfx_average_along_frames)
                dist_pre_eq = Decoder_Oscillator_Noise_FX.modified_sigmoid(dist_pre_eq)
                dist_ctanh = self.dense_zfx_average_to_dist_ctanh_knob (latent_zfx_average_along_frames)# One Scalar per audio. Can Have Range 0~5
                if self.config.distortion.ctanh_sigmoid == True: #Sigmoid ensures ctanh to be positive
                    pre_amp_before_tanh =  5 * Decoder_Oscillator_Noise_FX.modified_sigmoid(dist_ctanh)[..., -1]
                    dist_ctanh =  5 * Decoder_Oscillator_Noise_FX.modified_sigmoid(dist_ctanh)[..., :-1]
                    # dist_ctanh = 5 * sigmoid(dist_ctanh) -> Limit Range of dist_ctanh -inf ~ +inf into positive number 0~5
                    # If I limit range of coeffcient of each tanh term 0~1, It is hard to realize Post-Amplifying effect of Distortion
            else:
                # Produce Time-Varying adujustable knob values for W_H_distortion Function. (Get FX Knob varying by temporal frames) #
                dist_pre_eq = self.dense_zfx_framewise_to_dist_pre_eq_knob (gru_applied_latent_total_dry)
                dist_pre_eq = Decoder_Oscillator_Noise_FX.modified_sigmoid(dist_pre_eq)
                dist_ctanh = self.dense_zfx_framewise_to_dist_ctanh_knob (gru_applied_latent_total_dry)
                if self.config.distortion.ctanh_sigmoid == True: #Sigmoid ensures ctanh to be positive
                    pre_amp_before_tanh =  5 * Decoder_Oscillator_Noise_FX.modified_sigmoid(dist_ctanh)[..., -1]
                    dist_ctanh =  5 * Decoder_Oscillator_Noise_FX.modified_sigmoid(dist_ctanh)[... , :-1]
                    # dist_ctanh = 5 * sigmoid(dist_ctanh) -> Limit Range of dist_ctanh -inf ~ +inf into positive number 0~5
                    # If I limit range of coeffcient of each tanh term 0~1, It is hard to realize Amplifying effect of Distortion
            print("Time Consuming for Decoding is : ", time.time()-start_dec)
            return dict(one_f0=batch["one_f0"], a=a_1st, c=c_1st, H=H, dist_pre_eq = dist_pre_eq, dist_ctanh = dist_ctanh, pre_amp_before_tanh = pre_amp_before_tanh)
        else: 
            print("Time Consuming for Decoding is : ", time.time()-start_dec)
            return dict(one_f0=batch["one_f0"], a=a_1st, c=c_1st, H=H)
    @staticmethod
    def modified_sigmoid(a):
        a = a.sigmoid()
        a = a.pow(2.3026)  # log10
        a = a.mul(2.0)
        a.add_(1e-7)
        return a