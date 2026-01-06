# download the pipeline from Huggingface
from pyannote.audio import Pipeline
import torch
import os

HUGGINGFACE_TOKEN=os.getenv("HUGGINGFACE_TOKEN")
device = "cuda" if torch.cuda.is_available() else "cpu"

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1", 
    use_auth_token=HUGGINGFACE_TOKEN).to(device) # send pipeline to GPU (when available)
# run the pipeline locally on your computer
output = pipeline(file="./data/audio_de_teste.wav", num_speakers=3)

# print the predicted speaker diarization 
for turn, speaker in output.speaker_diarization:
    print(f"{speaker} speaks between t={turn.start:.3f}s and t={turn.end:.3f}s")
