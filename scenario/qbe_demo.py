import os
import time
import queue
import pydub
import random 

import numpy as np
import soundfile as sf
import streamlit as st
import base64
from auditok.signal import calculate_energy_single_channel
from typing import List
from datetime import datetime
from collections import deque
from itertools import groupby
from operator import itemgetter
from streamlit.proto.Markdown_pb2 import Markdown
from streamlit_webrtc import (
    ClientSettings,
    WebRtcMode,
    webrtc_streamer,
)

from utils.helper import * 
import torch
from scenario.qbe import *



THRESHOLD = 0.5
MIN_FRAMES = 2
PADDING_NUMBER = 10
WINDOW_SIZE =  1.0 # second
WINDOW_STEP = 0.25 # second
SAMPLE_RATE = 16000

FOLDER_PATH = "./data"
CKPT_PATH = "./ckpt/bcres_softmax_ce_12_aug/checkpoints/model.ckpt"
# CKPT_PATH = "./ckpt/resnet_arcface_ce_35_0.4_10_0001_94/checkpoints/model.ckpt"
file_ = open("./asset/voice3.gif", "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()

def app_kws(
            average_embed,
            threshold,
            sample_rate: int=16000, 
            window_size: int=1, 
            window_step: float=0.25):

    webrtc_ctx = webrtc_streamer(
        key="keyword-spotting",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        client_settings=ClientSettings(
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
            media_stream_constraints={"video": False, "audio": True},
        ),
    )

    status_indicator = st.empty()
    all_sound = pydub.AudioSegment.empty()
    processed_sound = pydub.AudioSegment.empty()
    count = 0
    active_ids = []
    text_out_list = []

    if not webrtc_ctx.state.playing:
        return

    status_indicator.write("Loading...")

    st.markdown(
    f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
    unsafe_allow_html=True,
)
    keyword = st.empty()

    while True:
        if webrtc_ctx.audio_receiver:
            try:
                audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
            except queue.Empty:
                time.sleep(0.1)
                status_indicator.write("No frame arrived.")
                continue
            
            status_indicator.write("Running. Say something!")

            for audio_frame in audio_frames:
                sound = pydub.AudioSegment(
                    data=audio_frame.to_ndarray().tobytes(),
                    sample_width=audio_frame.format.bytes,
                    frame_rate=audio_frame.sample_rate,
                    channels=len(audio_frame.layout.channels),
                )
                sound = sound.set_channels(1).set_frame_rate(sample_rate)
                
                all_sound += sound
                processed_sound += sound
                
            idx_pred = 0
            if len(all_sound) - count * window_step * 1000 >= window_size * 1000:
                now = datetime.now()
                sound_chunk = all_sound[count*1000*window_step: count*1000*window_step + window_size*1000]
                sound_chunk_to_list = sound_chunk.get_array_of_samples().tolist()
                sound_chunk_energy = calculate_energy_single_channel(np.array(sound_chunk_to_list, dtype=np.int16), 2)

                print('='*100)
                print('SOUND CHUNK ENERGY: {}'.format(sound_chunk_energy))
                print(f"COUNT: {count}")
                print(f"ALL-SOUND: {len(all_sound)} - {all_sound.duration_seconds}")
                print(f"SOUND-CHUNK:  {sound_chunk.duration_seconds} [{count*1000*window_step}: {count*1000*window_step + window_size*1000}]")
                print('='*100)
                print(type(sound_chunk_to_list))
                # st.session_state.keyword = st.text("NON-KEYWORD")
                st.session_state.keyword = "NON-KEYWORD"
                keyword.markdown("=========================")
                if sound_chunk_energy > 55:

                    sound_chunk_torch = torch.unsqueeze(torch.Tensor(sound_chunk_to_list),0)
                    print(sound_chunk_torch.shape)
                    pad_audio = padding(sound_chunk_torch)
                    x = pad_audio.to(args.device)
                    
                    inf_embed,inf_embed_torch = init_embedding(model,args,x)
                    cos = torch.nn.CosineSimilarity(dim=0)
                    d = cos(average_embed, inf_embed_torch) 
                    print("thres", threshold)
                    print("distance",d)
                    # keyword = st.text("NON-KEYWORD")
                    if d>=threshold:
                        idx_pred=1
                        st.session_state.keyword = "KEYWORD"
                        keyword.markdown("======== KEYWORD ========")
                        timestamp = count*1000*window_step
                        timestamp = int(timestamp)

                
                text_out_list.append(int(idx_pred))


                if int(idx_pred) == 1:
                    # sound_chunk.export(f'./out{str(count)}.wav', format='wav')
                    active_ids.append(count)
                


                # TODO: Update variable: count
                count += 1

        else:
            status_indicator.write("AudioReciver is not set. Abort.")
            break

def lit_audio(place_holder_raw_audio, sound, TEMP_DIR):
    try:
        temp_path = os.path.join(TEMP_DIR, f'raw_audio.wav')
        sound.export(temp_path, format='wav')

        audio_file = open(temp_path, 'rb')
        audio_bytes = audio_file.read()
        place_holder_raw_audio.audio(audio_bytes, format='audio/wav')
    except:
        st.markdown(
            "Error in lit_audio!"
        )


def lit_processed_audio(placeholder_processed_audio, sound, active_idx, TEMP_DIR):
    if len(active_idx) > 0:
        print(f"\n\t>>> TODO: ---------- LIT PROCESS AUDIO ----------")
        temp_path = os.path.join(TEMP_DIR, f'processed_audio.wav')

        ranges = find_consecutive_values(data=active_idx)
        array_sound = np.array(sound.get_array_of_samples())
        array_sound = np.expand_dims(array_sound, axis=0)

        print(f"-> ACTIVE INDEXES: {active_idx} {ranges} | ARRAY_SOUND: {np.shape(array_sound)}")

        ## TODO: replace bip

        for r, irange in enumerate(ranges):
            
            start = int(irange[0] * SAMPLE_RATE * WINDOW_STEP)
            end = int(irange[1] * SAMPLE_RATE * WINDOW_STEP + 16000)
            
            if array_sound.shape[1] > end:
                start = start - PADDING_NUMBER
                end = end + PADDING_NUMBER

            print(f"- STart: {start} | ENd: {end}")    
            beep_r = np.tile(wav_beep, (1, irange[1] + 2 - irange[0]))
            beep_r = beep_r[0, :end - start]
            print(f"- BEEP_R: {np.shape(beep_r)} | ARRAY_SOUND CHUNK: {np.shape(array_sound[0, start: end])}")
            beep_r = np.expand_dims(np.array(beep_r), axis=0)

            array_sound[0, start: end] = beep_r[0, :]

        processed_sound = sound._spawn(array_sound)
        ## TODO: export to file 
        processed_sound.export(temp_path, format='wav')

        ## TODO: Show in st
        audio_file = open(temp_path, 'rb')
        audio_bytes = audio_file.read()
        placeholder_processed_audio.audio(audio_bytes, format='audio/wav')


def find_consecutive_values(data):
    ranges =[]

    for k,g in groupby(enumerate(data),lambda x:x[0]-x[1]):
        group = (map(itemgetter(1),g))
        group = list(map(int,group))
        if group[-1] - group[0] >= MIN_FRAMES - 1:
            ranges.append((group[0],group[-1]))

    return ranges




"# Keyword Spotting Demo"


model_load_state = st.text("Loading pretrained models...")

args = get_args()
args.model = 'bcres'
args.metric = 'softmax'
args.n_keyword = 12
model = load_model(CKPT_PATH,args)


beep_file = 'beep.wav'
beep_ = pydub.AudioSegment.from_wav(beep_file)
beep_ = beep_.set_channels(1).set_frame_rate(SAMPLE_RATE)
# beep_ = beep_[:1000]
wav_beep = beep_.get_array_of_samples()


st.header("1. Record your own voice")
# keyword = st.text_input('Enter your keyword','')
col1, col2 = st.columns([1,1])
with col1: 
    record_btn = st.button(f"Click to Record")
with col2: 
    reset_btn = st.button(f"Reset")
if reset_btn: 
    for f in os.listdir("./data/keyword/"):
        os.remove(os.path.join("./data/keyword/",f))


if record_btn:
    # if keyword == "":
    #     st.warning("Choose a filename.")
    # else:
    record_state = st.text("Recording...")
    duration = 1.5  # seconds
    fs = 16000
    myrecording = record(duration, fs)
    
    record_state.text(f"Saving sample of the keyword ")
    num_random = random.randint(0,50)
    num_random2 = random.randint(60,100)
    path_myrecording = f"./data/keyword/{num_random}_{num_random2}.wav"
    if os.path.isfile(path_myrecording):
        num_random = random.randint(num_random,num_random2)
        path_myrecording = f"./data/keyword/{num_random}_{num_random2}.wav"
    save_record(path_myrecording, myrecording, fs)
    record_state.text(f"Done! Saved sample")

    st.audio(read_audio(path_myrecording))

"## 2. Compute Embedding"
btn = st.button("Compute")
if 'average' not in st.session_state: 
    st.session_state.average = torch.zeros(512)

if 'embed_torch' not in st.session_state: 
    st.session_state.embed_torch = torch.zeros(512)

if btn==True: 
    compute_state = st.text("Computing")
    st.session_state.average, st.session_state.embed_torch = average_embedding(FOLDER_PATH,"keyword",CKPT_PATH,args)
    compute_state.text("Done!")

"## 3. Choose Threshold"

st.session_state.threshold = st.slider('Threshold', min_value=0.0, max_value=1.0, value=0.7, step=0.05)
if 'threshold' not in st.session_state: 
    st.session_state.threshold = 0.7


"## 4. Demo QbE"

st.text("Click START")

 
app_kws(
    average_embed=st.session_state.average,
    threshold = st.session_state.threshold,
    sample_rate=SAMPLE_RATE,
    window_size=WINDOW_SIZE,
    window_step=WINDOW_STEP
)