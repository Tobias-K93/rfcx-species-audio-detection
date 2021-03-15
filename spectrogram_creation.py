# ########## Dataset creation ##########
import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import torch

try:
    # local
    source_path = '/home/tobias/kaggle_competitions/rainforest_connection'
    train_tp = pd.read_csv(os.path.join(source_path, 'train_tp.csv'))
    sample_submission = pd.read_csv(os.path.join(source_path,
                                                 'sample_submission.csv'))
except FileNotFoundError:
    # kaggle
    source_path = '../input/rfcx-species-audio-detection'
    train_tp = pd.read_csv(os.path.join(source_path, 'train_tp.csv'))
    sample_submission = pd.read_csv(os.path.join(source_path,
                                                 'sample_submission.csv'))

os.mkdir('spectrogram')

# Choose type #############
make_mel_spectrogram = True
affix = 'mel'

# creating dataset by applying short-time furior transform and save it as
# pytorch tensors
sampling_rate = 2**15
frame_size = 2048  # 2184
window_size = frame_size
hop_length = window_size//4
num_mels = 240

for i, id in enumerate(train_tp['recording_id']):  # sample_submission
    audio_file = os.path.join(source_path, 'train', f'{id}.flac')  # 'test'
    y, sr = librosa.load(audio_file, sr=sampling_rate)
    y_tensor = torch.tensor(y)
    torch.save(y_tensor, os.path.join('audio_tensors', f'{id}.pt'))
    if make_mel_spectrogram:
        # output dimension: (number of mels, frames)
        y_stft = librosa.feature.melspectrogram(y,
                                                n_fft=frame_size,
                                                hop_length=hop_length,
                                                win_length=window_size,
                                                sr=sampling_rate,
                                                n_mels=num_mels)
    else:
        # dimension: (frequency bins (y-axis/frequency), frames (x-axis/time))
        y_stft = librosa.stft(y, n_fft=frame_size,
                              hop_length=hop_length,
                              win_length=window_size)
    y_db = librosa.amplitude_to_db(np.abs(y_stft))
    y_db_tensor = torch.from_numpy(y_db)

    torch.save(y_db_tensor, os.path.join('spectrogram', f'{id}_{affix}.pt'))

    if i % 100 == 0:
        print(f"Progress: {round(i/len(train_tp['recording_id'])*100, 1)}%")
        # sample_submission
