import os
from datetime import datetime
import random 
from aiortc.contrib.media import MediaRecorder
from streamlit_webrtc import webrtc_streamer
from streamlit_webrtc import (
    ClientSettings,
    WebRtcMode,
    webrtc_streamer,
)

# create a function to record voice of user 
def record(keyword):
    num_random = random.randint(0,50)
    num_random2 = random.randint(60,100)
    path = f"./data/{keyword}/{keyword}_{num_random2}_{num_random}.wav"
    # create a function to format voice input files
    def in_recorder_factory() -> MediaRecorder:
        dtime = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
        num_random = random.randint(0,50)
        path = f"./data/keyword/{keyword}_{dtime}_{num_random}_input.wav"
        return MediaRecorder(path, format="wav")
    # create a function to format voice output files
    def out_recorder_factory() -> MediaRecorder:
        dtime = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
        num_random = random.randint(0,50)
        num_random2 = random.randint(60,100)
        if not os.path.isdir(f"./data/{keyword}/"):
            os.mkdir(f"./data/{keyword}/")
        # path = f"./data/{keyword}/{keyword}_{num_random2}_{num_random}.wav"
        return MediaRecorder(path, format="wav")
    # setup web streamer 
    webrtc_streamer(
        key="loopback",
        mode=WebRtcMode.SENDRECV,
        client_settings=ClientSettings(
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={
                "video": False,
                "audio": True,
            },
        ),
        # in_recorder_factory=in_recorder_factory,
        out_recorder_factory=out_recorder_factory,
        sendback_audio=True
    )
    return path 
    


# if __name__ == "__main__":
#     app()