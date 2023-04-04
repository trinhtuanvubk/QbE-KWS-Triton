import os
import matplotlib.pyplot as plt
import numpy as np 
from scipy.io import wavfile
import sounddevice as sd
import wavio
import tqdm 
from time import sleep
from tqdm import tqdm 
import streamlit as st

import librosa
from pathlib import Path
import sounddevice as sd
import wavio


# create a function to map each words with a number 
def mapping_words(list_path): 
    with open(list_path,'r') as f_in : 
        lines = f_in.readlines()
        dicts = {}
        for count,l in enumerate(lines) : 
            dicts[l.strip()] = count
            if not os.path.isdir('./data/' + str(count)):
                os.mkdir('./data/'+ str(count))
    f_in.close()
    return dicts

#  create a function to draw spectrogram 
def draw_spectrogram(file_path) : 
    # Read the wav file 
    samplingFrequency, signalData = wavfile.read(file_path)
    # Plot the signal read from wav file
    plt.title('Spectrogram of your wav file')
    plt.plot(signalData)
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.specgram(signalData,Fs=samplingFrequency)

def create_spectrogram(voice_sample):
    """
    Creates and saves a spectrogram plot for a sound sample.
    Parameters:
        voice_sample (str): path to sample of sound
    Return:
        fig
    """

    in_fpath = Path(voice_sample.replace('"', "").replace("'", ""))
    original_wav, sampling_rate = librosa.load(str(in_fpath))

    # Plot the signal read from wav file
    fig = plt.figure()
    plt.subplot(211)
    # plt.title(f"Spectrogram of file {voice_sample}")

    plt.plot(original_wav)
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")

    # plt.subplot(212)
    # plt.specgram(original_wav, Fs=sampling_rate)
    # plt.xlabel("Time")
    # plt.ylabel("Frequency")
    # plt.savefig(voice_sample.split(".")[0] + "_spectogram.png")
    return fig

def read_audio(file):
    with open(file, "rb") as audio_file:
        audio_bytes = audio_file.read()
    return audio_bytes

def record(duration=5, fs=48000):
    sd.default.samplerate = fs
    sd.default.channels = 1
    myrecording = sd.rec(int(duration * fs))
    # pbar = tqdm(total=100)
    # for i in range(10):
    #     sleep(0.1)
    #     pbar.update(10)
    # pbar.close()
    sd.wait(duration)
    return myrecording

def save_record(path_myrecording, myrecording, fs):
    wavio.write(path_myrecording, myrecording, fs, sampwidth=2)
    return None


    



