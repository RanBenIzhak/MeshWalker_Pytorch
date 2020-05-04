import torch
from torch import nn
from easydict import EasyDict as ed
import numpy as np

class RnnWalkBase(nn.Module):
    def __init__(self, params):
        super(RnnWalkBase, self).__init__()
        self._classes = params.n_classes
        self._params = params
        self._input_dim = params.number_of_features
        self._add_skip_connections = 'skip' in self._params.aditional_network_params
        self._pooling_betwin_grus = 'pooling' in self._params.aditional_network_params

        self._init_model()

    def _init_model(self):
        raise NotImplementedError('Must implement in children classes')

    def _print_fn(self, st):
        with open(self._params.logdir + '/log.txt', 'at') as f:
            f.write(st + '\n')


class RnnWalkNet(RnnWalkBase):
    def __init__(self, params):
        if params.layer_sizes is None:
          self._layer_sizes = {'fc1': 128, 'fc2': 256, 'gru1': 1024, 'gru2': 1024, 'gru3': 512}
        else:
          self._layer_sizes = params.layer_sizes
        super(RnnWalkNet, self).__init__(params)

    def _init_model(self):
        # kernel_regularizer = tf.keras.regularizers.l2(0.0001)
        # initializer = tf.initializers.Orthogonal(3)
        self._use_norm_layer = self._params.use_norm_layer is not None
        if self._params.use_norm_layer == 'InstanceNorm':
            norm_layer = nn.InstanceNorm1d
        elif self._params.use_norm_layer == 'BatchNorm':
            norm_layer = nn.BatchNorm1d
        else:
            raise ValueError('Norm layer not recognized - {}'.format(self._params.use_norm_layer))

        self.fc1 = nn.Linear(self._input_dim, self._layer_sizes['fc1'])
        self.fc2 = nn.Linear(self._layer_sizes['fc1'], self._layer_sizes['fc2'])
        self.relu = nn.ReLU()
        self.norm1 = norm_layer(self._layer_sizes['fc1'])
        self.norm2 = norm_layer(self._layer_sizes['fc2'])

        inp_sizes = [self._layer_sizes['fc2'],
                     self._layer_sizes['gru1'],
                     self._layer_sizes['gru2'],
                     self._layer_sizes['gru3']]
        if self._add_skip_connections:
            if self._pooling_betwin_grus:
                inp_sizes = [256, 640, 896, 1792]
            else:
                inp_sizes = [256, 1024+256, 256 + 1024*2, 512]
        elif  self._pooling_betwin_grus:
            inp_sizes = [256, 512, 512, 512]


        self._gru1 = nn.GRU(inp_sizes[0], self._layer_sizes['gru1'], batch_first=True)
        self._gru2 = nn.GRU(inp_sizes[1], self._layer_sizes['gru2'], batch_first=True)
        self._gru3 = nn.GRU(inp_sizes[2], self._layer_sizes['gru3'], batch_first=True)
        self._fc_last = nn.Linear(inp_sizes[3], self._classes)

        self._dropout = nn.Dropout(p=self._params.net_gru_dropout)
        self._pooling = nn.MaxPool1d(kernel_size=3, stride=2, padding=1) # TODO: verify padding is as 'same'
        # layers.Dense(self._classes, activation=self._params.last_layer_actication, kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
        #                              kernel_initializer=initializer)

    def forward(self, x):
        '''

        :param self:
        :param x: [Seq_len, Batch_size, Features]
        :return:
        '''
        x = self.relu(self.norm1(self.fc1(x).permute(0,2,1)).permute(0,2,1))
        x = self.relu(self.norm2(self.fc2(x).permute(0, 2, 1)).permute(0, 2, 1))
        # x1, _ = self._gru1(self._dropout(x))
        x1, _ = self._gru1(x)

        if self._add_skip_connections:
            x1 = torch.cat([x, x1], dim=2)
        if self._pooling_betwin_grus:
            x1 = self._pooling(x1)
            if self._add_skip_connections:
                x = self._pooling(x)
        x2, _ = self._gru2(x1)
        if self._pooling_betwin_grus:
            x2 = self._pooling(x2)
            if self._add_skip_connections:
                x = self._pooling(x)
                x1 = self._pooling(x1)
        if self._add_skip_connections:
            x2 = torch.cat([x, x1, x2], dim=2)
        x3, _ = self._gru3(x2)
        if self._add_skip_connections:
            x3 = torch.cat([x, x1, x2, x3], dim=2)
        x = x3
        if self._params.one_label_per_model:
            out_vec = x[:, -1, :]

            predict_vec = self._fc_last(out_vec)
        return x, predict_vec


if __name__ == '__main__':
    print('debug')
    x = torch.randn(50,4,3)
    params = ed({'net_gru_dropout': 0.5,
                 'use_norm_layer': 'InstanceNorm',
                 'layer_sizes': None,
                 'aditional_network_params': ['skip', 'pooling'],
                 'one_label_per_model': True,
                })
    classes=10
    model = RnnWalkNet(params, classes, 3)

    out_states,pred_vec = model(x)
    print(out_states.shape)
