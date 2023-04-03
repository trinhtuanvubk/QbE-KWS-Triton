import time
import queue
import pydub
import tempfile

import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st
import base64
from auditok.signal import calculate_energy_single_channel
from typing import List
from datetime import datetime
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
from scenario.classify import *



THRESHOLD = 0.5
MIN_FRAMES = 2
PADDING_NUMBER = 10
WINDOW_SIZE =  1.0 # second
WINDOW_STEP = 0.25 # second
SAMPLE_RATE = 16000
WIDTH_PREDICTION = 10
WIDTH_LINE_CHART = 10
TEMP_DIR = tempfile.mkdtemp()
FOLDER_PATH = "./data"
CKPT_PATH = "./ckpt/bcres_softmax_ce_12_aug/checkpoints/model.ckpt"
file_ = open("./asset/voice3.gif", "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()

arr_12 = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go',
                     'unknown', 'silence']

arr_35 = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go',
                     'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
                     'bed', 'bird', 'cat', 'dog', 'happy', 'house', 'marvin', 'sheila', 'tree', 'wow',
                     'backward', 'forward', 'follow', 'learn', 'visual']

def app_kws(
            # average_embed,
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
                keyword.markdown("=====================")
                if sound_chunk_energy > 55:

                    sound_chunk_torch = torch.unsqueeze(torch.Tensor(sound_chunk_to_list),0)
                    print(sound_chunk_torch.shape)
                    pad_audio = padding(sound_chunk_torch)
                    x = pad_audio.to(args.device)
                    

                    label, probs = classify(model, args, x, arr_12)
                    # keyword = st.text("NON-KEYWORD")
                    if probs>=threshold:
                        idx_pred=1
                        # st.session_state.keyword.text("KEYWORD")
                        # st.session_state.keyword = "KEYWOR"

                        keyword.markdown("========= {} =========".format(label.capitalize()))
                        timestamp = count*1000*window_step
                        timestamp = int(timestamp)
                
                text_out_list.append(int(idx_pred))

                if int(idx_pred) == 1:
                    # sound_chunk.export(f'./out{str(count)}.wav', format='wav')
                    active_ids.append(count)
                # TODO: Update variable: count
                count += 1

            # TODO: View line chart
            LENGTH_FRAMES = len(text_out_list)
            if len(text_out_list) >= WIDTH_LINE_CHART:
                index = [i for i in range(LENGTH_FRAMES - WIDTH_LINE_CHART, LENGTH_FRAMES)]
                seconds = [str(window_step * i) for i in index]
                predictions = text_out_list[-WIDTH_LINE_CHART:]
            else:
                seconds = [str(window_step * i) for i in range(LENGTH_FRAMES)]
                predictions = text_out_list

            data = pd.DataFrame({
                'second': seconds,
                'predict': predictions
            })

            data = data.rename(columns={'second': 'index'}).set_index('index')
        

        else:
            status_indicator.write("AudioReciver is not set. Abort.")
            break

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
model = load_model(CKPT_PATH,args)


"## 1. Choose Threshold"

st.session_state.threshold = st.slider('Threshold', min_value=0.0, max_value=1.0, value=0.7, step=0.05)
if 'threshold' not in st.session_state: 
    st.session_state.threshold = 0.7


"## 2. Demo Classification"

st.text('YES | NO | UP | DOWN | LEFT | RIGHT | ON | OFF | STOP | GO | UNKNOW | SILENCE ')
    
st.text("Click START")


app_kws(
    threshold = st.session_state.threshold,
    sample_rate=SAMPLE_RATE,
    window_size=WINDOW_SIZE,
    window_step=WINDOW_STEP
)