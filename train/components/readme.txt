"Components" directory contains non-trainable modules that make up the GDDSP.
Trainable Neural Nets(=Encoder and Decoder) do not belong to "Components" directory.
Here "non-trainable" means that the functions' types(Sinusoids/Polynomial/exponential), weights, biases of these modules do not change during Training and Inference.
In other words, "The Mapping Mechanisms" between Input and Output of these Components are Fixed.
However, the output audible signals of these Modules are not fixed, still can vary.
It's because the Independent Variables(=Input Arguments) of these moudules which are produced by decoder can vary by input audio.


- Harmonic Oscillator
    In GDDSP, The trainable decoder maps the polyphonic-f0, z ,loudness into Amplitude, phase ,number of harmonics.
    Then the harmonic Oscillator utilizes those outputs to make multiple sinusoids, which is the base of the dry sound.

- Noise Generator
    Just like Oscillators, the Noise Generator also need polyphonic_f0, z, loudness as input.
    The difference is that the Noise Generator reconstructs the non-periodic dry sound of input audio

- FX chain
    The FX modules are also included in this folder.
    The mathematical mechanism that each FX uses is fixed.
    Still, there are some adjustable knob values for each FX module, and those knob values are given by the decoder.

    - FX List
        - distortion
        - delay
        - reverb

    FX chain is used at two objectives in PolyPhonic Guitar DDSP - experiment.
    
    Firstly, It is used to produce Wet-GuitarSet. 
    FX Chain with Random Knob Values will render Effects on 360 audio samples in GuitarSet, and dataset.py will slice those samples into 4-sec excerpts.
    
    Secondly, It is used as the part of Synthesizing Modules of Guitar DDSP.
    The Data-Processing Steps of Guitar DDSP is same as below.
    Step1. Audio Input -> Fed to Encoder -> Converted to {Six stages of f0, loudness, z}
    Step2. {Six stages of f0, loudness, z} -> Fed to Decoder -> Converted to {Amplitudes and Phases of Sinusoids, Noise Band Coefficeint, FX Knob values}
    Step3. {Amplitudes and Phases of Sinusoids, Noise Band Coefficeint} -> Fed to Non-trainable Oscillators and Noise Generator -> Converted to Reconstructed_Dry_Signal
    Step4. {Reconstructed Dry Signal, FX Knob Values} -> Fed to Non-Trainable FX-Chain layer -> Converted to Reconstructed_Wet_Signal
    In Step4, The FX Chain applies Audio Effects(Dist/Delay/Reverb) on batch_wise Dry signals, so that those signals become as similar as possible to Input Wet Audio Excerpts.