# Goal and the Function of GDDSP


# GOAL
Developing the Polyphonic-DDSP which can disentangle pitch-sequences, loudness-sequences, timbres, FX-parameters in Reference Auido recorded  by guitars.

# Model Features
1. The Model is DDSP-Like Model which can analyze/copy the audio components of reference audio. 

2. The Model differs from the original DDSP in that it can separately capture the clean timbre and the detailed information of FX chain(Distortion-Delay-Reverb).

3. Training of this model requires two-stage backpropations of spectral losses.
- The first Backpropation of Loss is triggered by Comparison between Clean Input and Clean Reconstruected audio at intermediate layer,
- The second Backpropation of loss is triggered by Comparison between wet Input and Wet Reconstructed audio at Final layer.
- The Six harmonic Oscillators and the noise generator reconstruct the clean audio.
- The reconstructed clean audio is fed into FX chain(DSP layers) and then coverted into wet reconstructed audio.

4. The Model can handle Polyphonic Audio. (Degree of Polyphony = 6)
The Original DDSP (https://github.com/magenta/ddsp) has one oscillator so that it only could generate one Pitch(f0) per one Temporal Frame.
However, a Guitar has six strings and is inherently a polyphonic instrument.
This is why Model contains six harmonic oscillators, which can generate at most Six simultaneous lines of melody.


# Two roles of the Model
1. To clone only FX-settings of input audio, you would take the values of parameters which are inferred by FX decoder and apply those values on your pedalboard.
2. To clone the full-timbre(clean+FX) of input audio and render that timbre on your notes, you would take the z-embedding(= the common condition used by Osillators and the FX-Chain) which represents all timbre-related informaion of input. In this case, the Model uses the sequence of {pitch, loudness} extracted from your recording while the harmonic structure and FX parameters still depend on the reference audio.



# Steps for Training
- There Must be "Multi-Stage Loss and Backpropation"
- Multi-Stage Lss means that there is at least one intermediate layer of which output will be compared with GT audio.

- Assume that there are both clean dataset and wet dataset and each audio sample is paired by filename.
(If there is "hex_pickup_Eb_rock_01_clean.wav", then there is also "hex_pickup_Eb_rock_01_wet.wav" and the latter is same as former + FX)
- Loss1 : Clean Loss = Spectral Loss(Clean reference Audio, Clean Reconstructed Audio)
    - Clean Reconstructed Audio = Output of the intermediate layer = 
                                = Output of six Harmonic Oscillators + Output of Noise Generator
- Loss2 : Wet Loss  =  Spectral Loss(Wet reference Audio, Wet Reconstructed Audio)
    - Wet Reconstructed Audio =  Output of the final layer =
                              =  Output of the FX_DSP_layers whose input is the clean reconstructed audio.

By these Multi-Stage losses, 
the harmnonic oscillators and noise-generators are leaded to capture only "Dry Timbre" without FX,
while the FX layers are leaded to capture "Effects(distortion,delay,reverb)" to convert clean sound into wet sound.


- Clean Dataset : Guitarset hexaphonic recordings _ debleeded  (https://zenodo.org/record/3371780)
- Wet Dataset : each sample from Clean Dataset + FX with Randomized Parameters
- Loss  :  Muiti-scale spectral loss between input and recon. (The mathematical formula of loss can be found in original DDSP paper).
