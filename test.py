import librosa

from pywhispercpp.model import Model

model = Model("base.en")
arr, _ = librosa.load(
    "/home/chenghao/workspace/stt/whisper/this_is_a_test.wav", sr=16000
)
segments = model.transcribe(arr, language="en", n_processors=1)
for segment in segments:
    print(segment.text)

# export LD_LIBRARY_PATH=/home/chenghao/miniforge3/envs/py312/lib/python3.12/site-packages:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH="/home/chenghao/miniforge3/envs/py312/lib/site-packages:${LD_LIBRARY_PATH}"
