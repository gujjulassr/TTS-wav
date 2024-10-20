import torch
import glow as glow
import os
from scipy.io.wavfile import write
import torch,sys
import numpy as np

from denoiser import Denoiser


MAX_WAV_VALUE = 32768.0
sampling_rate=22050
waveglow_path='/Users/samarasimhareddygujjula/Desktop/GameChanger/Vococers/waveglow/waveglow_checkpoint'


waveglow = torch.load(waveglow_path)['model']
waveglow = waveglow.remove_weightnorm(waveglow)
waveglow.eval()




denoiser_strength=0.0
is_fp16=False
sigma=0.6

file_path=sys.argv[1]

base_name=file_path.strip().split('.pt')[0]


print("Mel is the input here")

mel = torch.load(file_path).T
mel=torch.from_numpy(mel)
mel = (mel*5)-5
mel= np.power(10.0, mel)
mel = torch.log(mel)


print(mel)

mel = torch.autograd.Variable(mel)
mel = torch.unsqueeze(mel, 0)
mel = mel.half() if is_fp16 else mel
with torch.no_grad():
    audio = waveglow.infer(mel, sigma=sigma)
    if denoiser_strength > 0:
        audio = denoiser(audio, denoiser_strength)
    audio = audio * MAX_WAV_VALUE
audio = audio.squeeze()
audio = audio.cpu().numpy()
audio = audio.astype('int16')
audio_path = os.path.join(
    '/Users/samarasimhareddygujjula/Desktop/GameChanger/results/waveglow', f"{base_name}_synthesis.wav")

# Check if the file exists
if os.path.exists(audio_path):
    # Prompt the user for input
    choice = input(f"File {audio_path} already exists. Do you want to override it? (y/n): ").lower()
    
    if choice == 'y':
        print("Overriding the file...")
        # Code to override the file here
    else:
        print("Operation canceled.")
else:
    print(f"File does not exist. Proceeding with the operation...")

write(audio_path, sampling_rate, audio)
print(audio_path)