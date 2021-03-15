# ##### Rainforest Connection Species Audio Detection ######
import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt

source_path = ''  # 'rfcx-species-audio-detection'
# #### true positives ##############################
train_tp = pd.read_csv(os.path.join(source_path, 'train_tp.csv'))

# ## species counts ##############################
species_ids, id_counts = np.unique(train_tp['species_id'], return_counts=True)

species_ids_songtype_1, species_counts_songtype_1 = \
    np.unique(train_tp['species_id'][train_tp['songtype_id'] == 1],
              return_counts=True)
species_ids_songtype_4, species_counts_songtype_4 = \
    np.unique(train_tp['species_id'][train_tp['songtype_id'] == 4],
              return_counts=True)

# create arrays with all species ids that count song types 1 & 4 respectively
complete_species_ids_counts_songtype_1 = np.zeros(len(species_ids))
for id, count in zip(species_ids_songtype_1, species_counts_songtype_1):
    complete_species_ids_counts_songtype_1[id] = count

complete_species_ids_counts_songtype_4 = np.zeros(len(species_ids))
for id, count in zip(species_ids_songtype_4, species_counts_songtype_4):
    complete_species_ids_counts_songtype_4[id] = count


plt.bar(species_ids, complete_species_ids_counts_songtype_1, color='slateblue')
plt.bar(species_ids, complete_species_ids_counts_songtype_4, color='darkred',
        bottom=complete_species_ids_counts_songtype_1)
plt.ylabel('Count')
plt.xlabel('Species IDs')
plt.xticks(np.arange(24))
plt.legend(['songtype 1', 'songtype 4'])
plt.title('Species count (with songtypes)')
plt.show()

# ## min/max frequencies of annotated signal ##############################
# sort min
f_min_sort_ids = np.argsort(train_tp['f_min'])
min_frequency_sorted = train_tp['f_min'].sort_values()
max_frequency_sorted_min = train_tp['f_max'][f_min_sort_ids]

plt.plot(np.arange(len(min_frequency_sorted)),
         min_frequency_sorted,
         color='slateblue')
plt.plot(np.arange(len(min_frequency_sorted)),
         max_frequency_sorted_min,
         color='darkred')
plt.semilogy()
plt.ylim([90, 20000])
plt.ylabel('Frequency (log)')
plt.legend(['min frequency', 'max frequency'])
plt.grid(which='both', axis='y')
plt.title('Min and Max frequency (sorted min)')
plt.show()

# sort max
f_max_sort_ids = np.argsort(train_tp['f_max'])
max_frequency_sorted = train_tp['f_max'].sort_values()
min_frequency_sorted_max = train_tp['f_min'][f_max_sort_ids]

plt.plot(np.arange(len(min_frequency_sorted)),
         min_frequency_sorted_max, color='slateblue')
plt.plot(np.arange(len(min_frequency_sorted)),
         max_frequency_sorted, color='darkred')
plt.semilogy()
plt.ylim([90, 20000])
plt.ylabel('Frequency (log)')
plt.legend(['min frequency', 'max frequency'])
plt.grid(which='both', axis='y')
plt.title('Min and Max frequency (sorted max)')
plt.show()

# ## Range of frequencies ##############################
frequency_range = train_tp['f_max'] - train_tp['f_min']
frequency_range_sorted = np.sort(frequency_range)

plt.plot(frequency_range_sorted)
plt.ylabel('Frequency range')
plt.title('Frequency difference between min and max')
plt.show()

# for each species
fig, axes = plt.subplots(5, 5, sharex=False, sharey=True, figsize=[8, 8])
for i in range(5):
    for j in range(5):
        id = i*5 + j
        if id == 24:
            total_frequency_range = train_tp['f_max'] - train_tp['f_min']
            total_frequency_range_sorted = np.sort(total_frequency_range)
            axes[i, j].plot(total_frequency_range_sorted)
            plt.ylim([0, 10000])
            axes[i, j].set_title('All', fontsize=10)

        else:
            frequency_range_per_species = train_tp['f_max']\
                [train_tp['species_id'] == id] \
                - train_tp['f_min'][train_tp['species_id'] == id]
            frequency_range_per_species_sorted = \
                np.sort(frequency_range_per_species)
            axes[i, j].plot(frequency_range_per_species_sorted)
            plt.ylim([0, 10000])
            if id == 0:
                axes[i, j].set_title(f'\n\nID {id}', fontsize=10)
            else:
                axes[i, j].set_title(f'ID {id}', fontsize=10)
plt.suptitle('Frequency range per species')
plt.tight_layout()
plt.show()

# ## Lenght of annotated signals ##############################
annotated_signal_length = train_tp['t_max'] - train_tp['t_min']
annotated_signal_length_sorted = np.sort(annotated_signal_length)

plt.plot(np.arange(len(annotated_signal_length_sorted)),
         annotated_signal_length_sorted, color='darkred')
plt.ylabel('Seconds')
plt.ylim(bottom=0)
plt.grid(axis='y')
plt.title('Length of annotated signals (combined)')
plt.show()

# for each species
fig, axes = plt.subplots(5, 5, sharex=False, sharey=True, figsize=[8, 8])
for i in range(5):
    for j in range(5):
        id = i*5 + j
        if id == 24:
            axes[i, j].plot(annotated_signal_length_sorted, color='darkred')
            plt.ylim([0, 10])
            axes[i, j].set_title('All', fontsize=10)

        else:
            annotated_signal_length_per_species = train_tp['t_max']\
                [train_tp['species_id'] == id] \
                - train_tp['t_min'][train_tp['species_id'] == id]
            annotated_signal_length_per_species_sorted = \
                np.sort(annotated_signal_length_per_species)
            axes[i, j].plot(annotated_signal_length_per_species_sorted,
                            color='darkred')
            plt.ylim([0, 10])
            if id == 0:
                axes[i, j].set_title(f'\n\nID {id}', fontsize=10)
            else:
                axes[i, j].set_title(f'ID {id}', fontsize=10)
plt.tight_layout()
plt.suptitle('Length of annotated signals (in sec)')
plt.show()

# ### Librosa

row_id = 89
recording_id_example = train_tp['recording_id'].iloc[row_id]
audio_file = os.path.join(source_path, 'train', f'{"0a4e7e350"}.flac')
print(f"Species ID: {train_tp['species_id'].iloc[row_id]} | "
      f"Songtype: {train_tp['songtype_id'].iloc[row_id]}"
      f"\n\t\tSignal\nBeginning: {train_tp['t_min'].iloc[row_id]} | "
      f"End: {train_tp['t_max'].iloc[row_id]}"
      f"\nMin Freq: {train_tp['f_min'].iloc[row_id]} | "
      f"Max Freq: {train_tp['f_max'].iloc[row_id]}")


# play file
os.system(f'vlc {audio_file}')

sampling_rate = 2**15
y, sr = librosa.load(audio_file, sr=sampling_rate)

# seconds
cut_start = 35  # 50
cut_stop = 42.5  # 60

y_cut = y[int((len(y)/60)*cut_start):int((len(y)/60)*cut_stop)]

frame_size = 2048
# 2184 (hop2) -> 1801 frames
# 1638 (hop2)/3276 (hop4) -> 2401
window_size = frame_size
hop_length = window_size//4

# melspectrogram
y_mel = librosa.feature.melspectrogram(y_cut,
                                       n_fft=frame_size,
                                       hop_length=hop_length,
                                       win_length=window_size,
                                       sr=sampling_rate,
                                       n_mels=240,
                                       fmin=0,
                                       fmax=16000)

print(y_mel.shape)
y_mel_db = librosa.amplitude_to_db(y_mel)

librosa.display.specshow(y_mel_db.reshape(256, -1), sr=sampling_rate,
                         hop_length=hop_length,
                         x_axis='time', y_axis='linear')
plt.colorbar(format="%+2.f dB")

# #### false positives ##############################
train_fp = pd.read_csv('train_fp.csv')

species_ids_fp, id_counts_fp = np.unique(train_fp['species_id'],
                                         return_counts=True)
