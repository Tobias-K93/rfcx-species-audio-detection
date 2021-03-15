# ########## Single Training ########## #
import os
import multiprocessing
import time
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchaudio.functional import compute_deltas

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import label_ranking_average_precision_score as lrap_score
from sklearn.preprocessing import OneHotEncoder

try:
    # local
    from efficientnet_pytorch import EfficientNet
    from resnest.torch import resnest50
except ModuleNotFoundError:
    # kaggle
    os.system('pip install efficientnet_pytorch')
    os.system('pip install resnest --pre')
    from efficientnet_pytorch import EfficientNet
    from resnest.torch import resnest50  # , resnest101

num_cpus = multiprocessing.cpu_count()

# loading ids and labels
spectrograms_path = os.path.join('rfcx-240mel-spectrogram-fs2048',
                                 'spectrogram')
# 'rfcx-240mel-spectrogram-fs2048'
# 'rfcx-300mel-spectrogram-fs2184'

try:
    # local
    source_path = ''
    train_tp = pd.read_csv(os.path.join(source_path,
                                        'rfcx-species-audio-detection',
                                        'train_tp.csv'))
except FileNotFoundError:
    # kaggle
    source_path = '../input/'
    train_tp = pd.read_csv(os.path.join(source_path,
                                        'rfcx-species-audio-detection',
                                        'train_tp.csv'))


# ########## Classes and functions ########## #
class STFTDataset(torch.utils.data.Dataset):
    def __init__(self, recording_ids, labels, time_interval_starts=None,
                 time_interval_ends=None, mode=None, random_cropping_prob=0,
                 num_snippets=6, normalize=True, num_channels=3, deltas=True):
        self.labels = labels
        self.recording_ids = recording_ids
        self.normalize = normalize
        self.num_channels = num_channels
        self.time_interval_starts = time_interval_starts
        self.time_interval_ends = time_interval_ends
        self.mode = mode  # 'train'
        self.rdm_cropping_prob = random_cropping_prob  # 0.0-1.0
        self.rng = np.random.default_rng()
        self.num_snippets = num_snippets
        self.deltas = deltas

    def __len__(self):
        return len(self.recording_ids)

    def min_max_normalization(self, tensor, min=-80, max=80):
        return (tensor-min)/(max-min)

    def in_range(self, x, a, b):
        '''Test whether x is within range (a,b)'''
        return (x > a) & (x < b)

    def __getitem__(self, index):
        '''
        Generates one sample of data
        In "train" mode: Select 60/num_snippets sec cropped from 60 sec
        spectrogram. Otherwise return whole 60 sec
        '''
        rec_id = self.recording_ids[index]
        snip_num = self.num_snippets
        snip_length = 60/snip_num

        X = torch.load(os.path.join(source_path, spectrograms_path,
                                    f'{rec_id}_mel.pt'))
        y = torch.FloatTensor(self.labels[rec_id])

        if self.mode == 'train':
            rnd_cropping = bool(self.rng.binomial(1, self.rdm_cropping_prob))

            t_start = self.time_interval_starts[index]
            t_end = self.time_interval_ends[index]
            t_length = t_end - t_start
            # cut off last frame to get even number (e.g. 1921 -> 1920)
            num_time_frames = X.shape[-1]
            # snippet length (in seconds) equivalent in frames
            # e.g. 1920 frames / 6 snippets = 320 frames/snippet
            frames_snip_length = int(num_time_frames/snip_num)

            # randomly cropping snippet length sec from spectrogram and
            # adjust labels if necessary
            if rnd_cropping:
                crop_start = random.uniform(0, 60-snip_length)
                start_frame_index = int(crop_start / 60 * num_time_frames)
                end_frame_index = start_frame_index + frames_snip_length
                buffer = t_length*0.1
                # adjusting labels
                if not (self.in_range(t_start, crop_start,
                                      crop_start + snip_length - buffer)
                        or self.in_range(t_end, crop_start + buffer,
                                         crop_start + snip_length)):
                    y = torch.zeros_like(y)
            # cropping snippet length seconds around given time interval
            # [t_start, t_end]
            else:  # no random cropping
                if t_length < snip_length:
                    # avoiding cropping over limits (0sec/60sec)
                    max_moving_range = min(t_start, snip_length - t_length)
                    min_moving_range = max(0, t_start - (60-snip_length))
                    applied_moving_range = random.uniform(min_moving_range,
                                                          max_moving_range)
                    start_frame_index = int((t_start - applied_moving_range)
                                            / 60 * num_time_frames)
                    end_frame_index = start_frame_index + frames_snip_length
                else:
                    max_moving_range = t_length - snip_length
                    min_moving_range = 0
                    applied_moving_range = random.uniform(min_moving_range,
                                                          max_moving_range)
                    start_frame_index = int((t_start + applied_moving_range)
                                            / 60 * num_time_frames)
                    end_frame_index = start_frame_index + frames_snip_length

            X = X[:, start_frame_index:end_frame_index]

        if self.deltas and self.num_channels == 3:
            deltas_1 = compute_deltas(X)
            deltas_2 = compute_deltas(deltas_1)
            X = torch.stack([X, deltas_1, deltas_2])
        else:
            X = torch.stack([X]*self.num_channels)

        if self.normalize:
            X = self.min_max_normalization(X)
        return X, y


class EfficientNetModel(nn.Module):
    def __init__(self, num_classes=24, efficient_net_type='b3'):
        super().__init__()
        '''
        Input resolutions:
        EfficientNetB0 	224
        EfficientNetB1 	240
        EfficientNetB2 	260
        EfficientNetB3 	300
        EfficientNetB4 	380
        '''
        eff_net_neurons = {'b0': 1280,
                           'b1': 1280,
                           'b2': 1408,
                           'b3': 1536,
                           'b4': 1792}
        self.eff_net = EfficientNet.from_pretrained('efficientnet-'
                                                    f'{efficient_net_type}',
                                                    include_top=False)
        # num of ouput neuron of efficient net
        hidden_size = eff_net_neurons[efficient_net_type]

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.2)
        # self.relu = nn.ReLU()
        # self.linear = nn.Linear(hidden_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.eff_net(x)
        x = self.flatten(x)
        x = self.dropout(x)
        # x = self.relu(self.linear(x))
        x = self.classifier(x)

        return x


class ResNeStModel(nn.Module):
    def __init__(self, num_classes=24):
        super().__init__()
        self.net = resnest50(pretrained=True)
        # self.net = resnest101(pretrained=True)  # larger alternative model
        last_layer_in_features = self.net.fc.in_features
        self.net.fc = torch.nn.Linear(in_features=last_layer_in_features,
                                      out_features=num_classes)

    def forward(self, x):
        x = self.net(x)
        return x


def train_fct(train_set, batch_size):
    model.train()
    data_loader = DataLoader(train_set,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=num_cpus)
    train_loss = 0
    train_predictions = []
    train_labels = []
    for i, (inputs, labels) in enumerate(data_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        # using automatic mixed precession (amp)
        with torch.cuda.amp.autocast(enabled=use_amp):
            output = model(inputs)
            loss = loss_fct(output, labels)
        loss = torch.mean(loss)
        scaler.scale(loss).backward()

        optimizer.step()
        train_loss += loss.item()

        batch_predictions = output.cpu().detach()
        train_predictions.append(batch_predictions)
        train_labels.append(labels.cpu())

    train_predictions = torch.cat(train_predictions, dim=0).numpy()
    train_labels = torch.cat(train_labels)

    lrap = lrap_score(train_labels, train_predictions)
    avg_train_loss = train_loss / len(data_loader)
    return avg_train_loss, lrap


def val_fct(val_set, batch_size, input_width, snip_num=6, overlap=1):
    '''
    val_set: dataset
    batch_size: validation batch size
                (train_batch_size//number of spectrogram snippets)
    input_width: resolution of input width (columns dimension)
    snip_num: Number of snippets that spectrogram inputs are cut in
    overlap: int indicating overlap of spectrogram snippets
             1 = no overlap, 2 ~ 50% overlap
    '''
    model.eval()
    data_loader = DataLoader(val_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_cpus)

    time_cutoff = input_width*snip_num
    # stride used to cut spectrograms into chunks for validation/prediction
    # e.g. (2400-300)/(8*2-1) = 140
    validation_stride = (time_cutoff-input_width)//(snip_num*overlap-1)

    val_loss = 0
    val_predictions = []
    val_labels = []
    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        # adjust for last (potentially shorter) batch
        batch_size = inputs.shape[0]

        # go over spectrogram to cut out parts,
        # possibly overlapping with stride < kernel_size
        inputs_unfold = F.unfold(inputs[:, :, :, :time_cutoff],
                                 kernel_size=input_width,
                                 stride=validation_stride)
        # assuring correct order within batch
        inputs_transposed = inputs_unfold.transpose(1, 2)
        # reshape from (val_batch_size, overlap*snip_num, -1) to
        # (train_batch_size, filter channels, input_dim[0], input_dim[1])
        inputs_final = inputs_transposed.reshape(batch_size
                                                 * snip_num
                                                 * overlap,
                                                 3, input_width, input_width)
        # do same for labels for dimensions to match
        labels_duplicated = torch.cat([labels]
                                      * snip_num
                                      * overlap, dim=1).view(snip_num
                                                             * overlap
                                                             * batch_size,
                                                             labels.shape[1])
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=use_amp):
                output = model(inputs_final)
                loss = loss_fct(output, labels_duplicated)
            mean_loss_over_classes = torch.mean(loss, dim=1)
            loss_predicted = torch.amin(
                                torch.stack(
                                    torch.chunk(mean_loss_over_classes,
                                                chunks=batch_size)),
                                dim=1)
            loss = torch.mean(loss_predicted)
            val_loss += loss.item()

            pred_per_chunk = output.cpu().detach()

            # snip_num chunks x 8 batch components => 8 predictions
            # get highest probability per class over all snip_num
            # spectrogram parts for each batch component
            batch_pred = torch.amax(torch.stack(torch.chunk(pred_per_chunk,
                                                            chunks=batch_size,
                                                            dim=0)),
                                    dim=1)
            val_predictions.append(batch_pred)
            val_labels.append(labels.cpu())

    val_predictions = torch.cat(val_predictions, dim=0)
    val_labels = torch.cat(val_labels)
    if make_songtype_extra:
        val_predictions[:, 17] = torch.amax(val_predictions[:, [17, 24]],
                                            dim=1)
        val_predictions[:, 23] = torch.amax(val_predictions[:, [23, 25]],
                                            dim=1)
        val_predictions = val_predictions[:, :24]

        val_labels[:, 17][val_labels[:, 24] == 1] = 1
        val_labels[:, 23][val_labels[:, 25] == 1] = 1
        val_labels = val_labels[:, :24]

    lrap = lrap_score(val_labels.numpy(), val_predictions.numpy())
    avg_val_loss = val_loss / len(data_loader)
    return avg_val_loss, lrap


def plot_cv(results, save_plot=False, close_plot=False, affix=''):
    '''
    Function to plot loss and metric of cross validation runs
    '''
    for j in range(2):
        fig, axes = plt.subplots(nrows=1,
                                 ncols=results.shape[0],
                                 figsize=[20, 4])
        for i in range(5):
            y_len_loss = min(len(results[i, 0, :][results[i, 0, :] != 10]),
                             len(results[i, 1, :][results[i, 1, :] != 10]))
            y_len_lrap = min(len(results[i, 2, :][results[i, 2, :] != 0]),
                             len(results[i, 3, :][results[i, 3, :] != 0]))

            if j == 0:
                axes[i].plot(range(1, y_len_loss + 1),
                             results[i, 0, :][:y_len_loss],
                             label='Train', color='slateblue')
                axes[i].plot(range(1, y_len_loss + 1),
                             results[i, 1, :][:y_len_loss],
                             label='Val', color='darkred')
                axes[i].set_xticks(range(1, y_len_loss + 1, 2))
                axes[i].legend()
                axes[i].set_xlabel('Epochs')
                axes[i].set_title(f'\nFold {i+1} ')

                plt.suptitle('Loss', fontsize=14)
                if i == 0:
                    axes[i].set_ylabel('Loss')
                plt.tight_layout()
                if i == 4:
                    plt.show()
                    if save_plot:
                        plt.savefig(f'cv_loss_plot{affix}')
                    if close_plot:
                        plt.close()
            else:
                axes[i].plot(range(1, y_len_lrap + 1),
                             results[i, 0+2, :][:y_len_lrap],
                             label='Train', color='slateblue')
                axes[i].plot(range(1, y_len_lrap + 1),
                             results[i, 1+2, :][:y_len_lrap],
                             label='Val', color='darkred')
                axes[i].set_xticks(range(1, y_len_lrap + 1, 2))
                axes[i].legend()
                axes[i].set_xlabel('Epochs')
                axes[i].set_title(f'\nFold {i+1} ')
                plt.suptitle('Label Ranking Average Precision Score',
                             fontsize=14)
                if i == 0:
                    axes[i].set_ylabel('LRAP Scoe')
                plt.tight_layout()
                if i == 4:
                    plt.show()
                    if save_plot:
                        plt.savefig(f'cv_lrap_plot{affix}')
                    if close_plot:
                        plt.close()


# ### Preprations for Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_model_weights = True

# whether examples with differeing songtypes (1 or 4) should be counted
# as 2 seperate classes during training
make_songtype_extra = False

use_deltas = True

recording_ids = train_tp['recording_id'].values.astype(str)
if make_songtype_extra:
    # change songtype id for label 16 from all 4 to all 1
    train_tp['songtype_id'][train_tp['species_id'] == 16] = 1
    # make species with songtype 4 extra species id class
    train_tp['species_id'][(train_tp['species_id'] == 17)
                           & (train_tp['songtype_id'] == 4)] = 24
    train_tp['species_id'][(train_tp['species_id'] == 23)
                           & (train_tp['songtype_id'] == 4)] = 25

labels = list(train_tp['species_id'])
one_hot_encoder = OneHotEncoder(sparse=False, dtype=np.float32)
labels_one_hot = one_hot_encoder.fit_transform(np.array(labels).reshape(-1, 1))
labels_dict = dict(zip(recording_ids, labels_one_hot))

time_interval_starts = train_tp['t_min'].values.astype(float)
time_interval_ends = train_tp['t_max'].values.astype(float)

seed = 123
num_classes = 26 if make_songtype_extra else 24
# number of snippets to cut spectrograms inputs in
snip_num = 8  # 6

early_stopping_patience = 10  # minimum 1
early_stopping_threshold = 0  # 1e-3
epochs = 36

eff_net_type = 'b2'
use_resnest = False

# ### Training
# training parameters
batch_size = 32
learning_rate = 0.0005
weight_decay = 0  # 1e-6
momentum = 0.1
rnd_crop_prob = 0.2
lr_plateau_factor = 0.1

scheduler_patience = 3
freeze_batchnorm = False
input_channels = 3

# validation parameters
overlap = 1
val_batch_size = batch_size // (snip_num * overlap)

loss_fct = nn.BCEWithLogitsLoss(reduction='none').to(device)

# ########## Cross Validation ########## #
results = []
torch.manual_seed(seed)
np.random.seed(seed)
strat_kfold = StratifiedKFold(n_splits=5, shuffle=True)
start_time = time.time()
for i, (train_ids, val_ids) in enumerate(strat_kfold.split(labels, labels)):
    print(f'\t-------------------- Fold {i+1} --------------------')
    train_set = STFTDataset(recording_ids[train_ids],
                            labels_dict,
                            time_interval_starts[train_ids],
                            time_interval_ends[train_ids],
                            mode='train',
                            random_cropping_prob=rnd_crop_prob,
                            num_channels=input_channels,
                            num_snippets=snip_num,
                            deltas=use_deltas)
    val_set = STFTDataset(recording_ids[val_ids], labels_dict,
                          num_channels=input_channels,
                          num_snippets=snip_num,
                          deltas=use_deltas)

    example_input = train_set.__getitem__(0)[0]
    input_width = example_input.shape[2]

    if use_resnest:
        model = ResNeStModel(num_classes=num_classes).to(device)
    else:
        model = EfficientNetModel(num_classes=num_classes,
                                  efficient_net_type=eff_net_type).to(device)
    if freeze_batchnorm:
        for child in next(model.children()).children():
            if isinstance(child, nn.BatchNorm2d):
                for param in child.parameters():
                    param.requires_grad = False

    # print('Model size: '
    #       f'{sum(layer.numel() for layer in model.parameters()):,}')

    # optimizer = torch.optim.Adam(model.parameters(),
    #                              lr=learning_rate,
    #                              weight_decay=weight_decay)
    optimizer = torch.optim.RMSprop(model.parameters(),
                                    lr=learning_rate,
                                    weight_decay=weight_decay,
                                    momentum=momentum)

    scheduler = ReduceLROnPlateau(optimizer,
                                  mode='min',  # 'max' for lrap
                                  factor=lr_plateau_factor,
                                  patience=scheduler_patience,
                                  threshold=0,  # 1e-3 for lrap
                                  verbose=True,)
    use_amp = device == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    train_loss_list = []
    val_loss_list = []
    train_lrap_list = []
    val_lrap_list = []
    patience_counter = 0
    for epoch in range(epochs):
        train_loss, train_lrap = train_fct(train_set, batch_size)
        val_loss, val_lrap = val_fct(val_set, val_batch_size,
                                     input_width, snip_num, overlap)
        # uncomment for epochwise tracking
        # print(f'Epoch {epoch+1:2} | Train Loss: {train_loss:.4}  \t'
        #       f'Validation Loss: {val_loss:.4}')
        # print(f'         | Train LRAP: {train_lrap:.3} \t'
        #       f'Validation LRAP: {val_lrap:.3}')

        # for early stopping (start counting aftering given epoch, here:9)
        if epoch < 10:
            pass
        elif (val_loss - min(val_loss_list)) > early_stopping_threshold:
            patience_counter += 1
        else:
            patience_counter = 0
            if save_model_weights:
                torch.save(model.state_dict(), f'state_dict_fold{i}_best')
        if save_model_weights:
            if (epoch % 5 == 0) & (epoch > 18):
                torch.save(model.state_dict(),
                           f'state_dict_fold{i}_epoch{epoch}')
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        train_lrap_list.append(train_lrap)
        val_lrap_list.append(val_lrap)

        # apply early stopping
        if patience_counter == early_stopping_patience:
            print(f'---- Early stopping with best val-lrap '
                  f'{np.max(val_lrap_list):.6} in epoch '
                  f'{np.argmax(val_lrap_list)+1} ----')
            break
        if epoch >= 10:
            scheduler.step(val_loss)

    best_val_lrap_fold = np.max(val_lrap_list)
    print(f'Fold {i+1} | Highest Validation LRAP: {best_val_lrap_fold:.4}')
    print('')

    train_losses = np.full((1, epochs), 10, dtype=np.float)
    val_losses = np.full((1, epochs), 10, dtype=np.float)
    train_lrap_scores = np.zeros((1, epochs))
    val_lrap_scores = np.zeros((1, epochs))

    train_losses[0, :len(train_loss_list)] = np.array(train_loss_list)
    val_losses[0, :len(val_loss_list)] = np.array(val_loss_list)
    train_lrap_scores[0, :len(train_lrap_list)] = np.array(train_lrap_list)
    val_lrap_scores[0, :len(val_lrap_list)] = np.array(val_lrap_list)

    results.append(np.concatenate((train_losses, val_losses,
                                  train_lrap_scores, val_lrap_scores),
                                  axis=0))

end_time = time.time()
training_time = round((end_time - start_time) / 60, 2)
print(f'Training Time: {training_time} min')

results = np.array(results)

best_losses = np.min(results[:, 1, :], axis=1)
best_lrap_scores = np.max(results[:, 3, :], axis=1)
oof_loss = np.mean(best_losses)
oof_lrap = np.mean(best_lrap_scores)
print(f'OOF LRAP Score: {oof_lrap:.4} | OOF Loss: {oof_loss:.6}')

final_scores = pd.DataFrame({'OOF loss': oof_loss,
                             'OOF LRAP': oof_lrap.round(4),
                             'Time': training_time}, [0])

plot_cv(results, save_plot=True, close_plot=True)

final_scores.to_csv('final_scores.csv', index=False)
