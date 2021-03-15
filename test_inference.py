# ######### Test Inference File ########## #
import os
import multiprocessing

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchaudio.functional import compute_deltas

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

test_spectrograms_path = 'rfcx-test-240mel-spectrogram-fs2048'
# 'rfcx-test-240mel-spectrogram-fs2048'
# 'rfcx-test-300mel-spectrogram-fs2184'

model_path = 'rfcx-single-model-training'

# loading ids and labels
try:
    # local
    source_path = ''
    sample_submission = pd.read_csv(
                            os.path.join(source_path,
                                         'rfcx-species-audio-detection',
                                         'sample_submission.csv'))
except FileNotFoundError:
    # kaggle
    source_path = '../input/'
    sample_submission = pd.read_csv(
                            os.path.join(source_path,
                                         'rfcx-species-audio-detection',
                                         'sample_submission.csv'))


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, recording_ids, normalize=True, num_channels=3,
                 deltas=True):
        self.recording_ids = recording_ids
        self.normalize = normalize
        self.num_channels = num_channels
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
        '''
        rec_id = self.recording_ids[index]

        X = torch.load(os.path.join(source_path,
                                    test_spectrograms_path,
                                    'spectrogram',
                                    f'{rec_id}_mel.pt'))  # _full

        if self.deltas and self.num_channels == 3:
            deltas_1 = compute_deltas(X)
            deltas_2 = compute_deltas(deltas_1)
            X = torch.stack([X, deltas_1, deltas_2])
        else:
            X = torch.stack([X]*self.num_channels)

        if self.normalize:
            X = self.min_max_normalization(X)

        return X


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

    def forward(self, inputs):
        x = self.eff_net(inputs)
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


def test_fct(test_set, batch_size, input_width, snip_num=8, overlap=1):
    model.eval()
    data_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_cpus)

    time_cutoff = input_width*snip_num
    # stride used to cut spectrograms into chunks for prediction
    # e.g. (2400-300)/(8*2-1) = 140
    test_stride = (time_cutoff-input_width)//(snip_num*overlap-1)

    test_predictions = []
    for inputs in data_loader:
        inputs = inputs.to(device)
        # adjust for last (potentially shorter) batch
        batch_size = inputs.shape[0]

        # go over spectrogram to cut out parts,
        # possibly overlapping with stride < kernel_size
        inputs_unfold = F.unfold(inputs[:, :, :, :time_cutoff],
                                 kernel_size=input_width,
                                 stride=test_stride)
        # assuring correct order within batch
        inputs_transposed = inputs_unfold.transpose(1, 2)
        # reshape from (val_batch_size, overlap*snip_num, -1) to
        # (train_batch_size, filter channels, input_dim[0], input_dim[1])
        inputs_final = inputs_transposed.reshape(batch_size
                                                 * snip_num
                                                 * overlap,
                                                 3, input_width, input_width)
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=use_amp):
                output = torch.sigmoid(model(inputs_final))
            pred_per_chunk = output.cpu().detach()

            # get highest probability per class over all spectrogram parts
            # for each batch component
            # e.g. 8 chunks x 4 batch components => 4 predictions
            batch_pred = torch.amax(torch.stack(torch.chunk(pred_per_chunk,
                                                            chunks=batch_size,
                                                            dim=0)),
                                    dim=1)
            test_predictions.append(batch_pred)

    test_predictions = torch.cat(test_predictions, dim=0)
    if make_songtype_extra:
        test_predictions[:, 17] = torch.amax(test_predictions[:, [17, 24]],
                                             dim=1)
        test_predictions[:, 23] = torch.amax(test_predictions[:, [23, 25]],
                                             dim=1)
        test_predictions = test_predictions[:, :24]

    return test_predictions


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = device == 'cuda'

test_recording_ids = sample_submission['recording_id'].values.astype(str)

eff_net_type = 'b3'  # 'b1'
input_channels = 3

make_songtype_extra = False

num_classes = 26 if make_songtype_extra else 24
snip_num = 8
overlap = 1
batch_size = 32
val_batch_size = batch_size // (snip_num * overlap)
input_width = 240
input_hight = input_width  # input_width  # 128

test_set = TestDataset(test_recording_ids, num_channels=input_channels)

model = EfficientNetModel(num_classes=num_classes,
                          efficient_net_type=eff_net_type).to(device)
# model = ResNeStModel(num_classes=num_classes).to(device)

test_predictions_list = []
for i in range(5):
    model.load_state_dict(
        torch.load(
            os.path.join(source_path,
                         model_path,
                         f'state_dict_fold{i}')))
    test_predictions_list.append(test_fct(test_set, val_batch_size,
                                          input_width, snip_num, overlap))

test_predictions = np.mean(torch.stack(test_predictions_list).numpy(), axis=0)
sample_submission.iloc[:, 1:] = test_predictions

sample_submission.to_csv('submission.csv', index=False)
