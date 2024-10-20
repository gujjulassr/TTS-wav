import os
import librosa


f=open('/raid/ai23mtech02001/GameChangerV2/Data/bengali_male_>20s.txt','a')

def get_wav_duration(wav_file):
    # Load the audio file with its original sample rate (sr=None keeps the original sample rate)
    audio, sr = librosa.load(wav_file, sr=None)
    # Get the duration of the audio
    duration = librosa.get_duration(y=audio, sr=sr)
    return duration

def calculate_total_duration(directory):
    total_duration = 0.0
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            file_path = os.path.join(directory, filename)
            duration = get_wav_duration(file_path)
            if duration>20:
                print(f"File: {filename}, Duration: {duration:.2f} seconds")
                full_path = os.path.join('/raid/ai23mtech02001/GameChangerV2/Data/kannada_female', file_path)
                f.write(f'{full_path}\n')
            total_duration += duration
    return total_duration

if __name__ == "__main__":
    folder_path = '/raid/ai23mtech02001/GameChangerV2/Data/bengali_male'
    total_duration = calculate_total_duration(folder_path)
    print(f"Total Duration of all .wav files: {total_duration:.2f} seconds")