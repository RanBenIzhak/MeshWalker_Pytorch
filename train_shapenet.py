from easydict import EasyDict
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
# =========== Paramaters ========= #
from utils.parameters import shapenet_params
params = shapenet_params('./weights/0438-25.04.2020..09.46__modelnet__CycLR-B_Xyz_rerun_faster')

# ========== Set Database and Dataloader ========= #
from datasets.ExactMatchChairs import ExactMatchChairs
exactmatch = ExactMatchChairs(params)
train_dl = DataLoader(exactmatch, batch_size=params.batch_size, shuffle=True, num_workers=4,
                      collate_fn=exactmatch.collate_fn, drop_last=False)
test_dl = DataLoader(exactmatch, batch_size=1, shuffle=False, num_workers=4,
                      collate_fn=exactmatch.collate_fn)
# ===== Set model ===== #
#currently only RnnWalkNet available - MeshWalker
from models.MeshWalker import RnnWalkNet
dnn_model = RnnWalkNet(params)

# Optional - load pretrained weights
if params.pretrained:
  # TODO: test if weights are keras weights
  net_files = os.listdir(params.pretrained)
  # TODO: assert required params (as input dimension, see if there are more) correspond to currnet params


# ====  set optimizer ==== #
dummy_param = []
if params.optimizer_type == 'adam':
  # TODO: pytorch needs network parameters sent to optimizer
  optimizer = torch.optim.Adam(params= dummy_param, lr = params.learning_rate[0])
  # TODO: add clipnorm - in backprop loop?
  clipnorm = params.gradient_clip_th
elif params.optimizer_type == 'cycle':
  # TODO: test cylic LR in pytorch
  # lr_schedule = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=params.cycle_opt_prms.initial_learning_rate,
  #                                                   maximal_learning_rate=params.cycle_opt_prms.maximal_learning_rate,
  #                                                   step_size=params.cycle_opt_prms.step_size,
  #                                                   scale_fn=lambda x: 1., scale_mode="cycle", name="MyCyclicScheduler")
  raise NotImplementedError('Need to implement cyclic optimizer')
elif params.optimizer_type == 'sgd':
  optimizer = torch.optim.SGD(dummy_param, lr=params.learning_rate[0], decay=0, momentum=0.9, nesterov=True, clipnorm=params.gradient_clip_th)
else:
  raise Exception('optimizer_type not supported: ' + params.optimizer_type)

# ==== Time stamps messages ==== #
time_msrs = {x: 0 for x in ['train_step', 'train_step_triplet', 'get_train_data', 'test']}

