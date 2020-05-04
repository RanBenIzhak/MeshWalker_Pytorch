import os

from easydict import EasyDict
import numpy as np

import utils.utils as utils
from utils import mesh, walks

from datasets import dataset_prepare
'''
Augmentations:
   arg_name       description                                     example
   --------       -----------                                     -------
- 'scaling'     - 2 numbers define scale range                    (0.5, 2)
- 'rotation'    - angle - +- the number supplied, for each side   30
- 'translation' - one number                                      2
- 'noise'       - std of normal noise to be added                 0.001
- 'stride'      - probability to skip a step                      0.2
'''


def set_up_default_params(network_task, run_name, cont_run_number=0):
  '''
  Define dafault parameters, commonly for many test case
  :param network_task: 'classification' or 'semantic_segmentation'
  :return: EasyDict with all parameters required for training
  '''
  params = EasyDict()

  params.cont_run_number = cont_run_number
  params.run_root_path = './runs/mesh_learning'
  params.logdir = utils.get_run_folder(params.run_root_path + '/rnn_mesh_walk' + '/', '__' + run_name, params.cont_run_number)
  params.model_fn = params.logdir + '/learned_model.keras'

  # Optimizer params
  params.optimizer_type = 'adam'  # sgd / adam
  params.learning_rate_dynamics = 'steps'
  params.learning_rate =       [2e-4, 1e-4,  5e-5,  2e-5 ]
  params.learning_rate_steps = [0,    50e3,  150e3, 300e3, np.inf]

  params.EPOCHS = 5000000
  params.n_models_per_test_epoch = 300

  params.gradient_clip_th = 1

  # Dataset params
  params.classes_indices_to_use = None
  params.train_dataset_size_limit = np.inf
  params.test_dataset_size_limit = np.inf
  params.network_task = network_task
  params.normalize_model = False
  params.sub_mean_for_data_augmentation = True
  params.datasets2use = {}
  params.test_data_augmentation = {}
  params.train_data_augmentation = {}
  params.aditional_network_params = []

  params.network_tasks = [params.network_task]
  if params.network_task == 'classification':
    params.n_walks_per_model = 1
    params.one_label_per_model = True
    params.train_loss = ['cros_entr']
  elif params.network_task == 'semantic_segmentation':
    params.n_walks_per_model = 4
    params.one_label_per_model = False
    params.train_loss = ['cros_entr']
  elif params.network_task == 'self:triplets':
    params.n_walks_per_model = 32
    params.one_label_per_model = False
    params.train_loss = ['triplet']
  elif params.network_task == 'semisupervised':
    params.n_walks_per_model = 32
    params.one_label_per_model = False
    params.train_loss = ['cros_entr', 'triplet']
    params.network_tasks = ['semantic_segmentation', 'self:triplets']
  else:
    raise Exception('Unsuported params.network_task: ' + params.network_task)
  params.batch_size = int(32 / params.n_walks_per_model)

  # Other params
  params.log_freq = 100
  params.seq_len = 100
  params.min_seq_len = int(params.seq_len / 2)
  params.walk_alg = 'no_jumps'   # no_repeat / no_jumps / fast / fastest / only_jumps / local_jumps / no_local_jumps
  params.net_input = ['dxdydz'] # 'xyz', 'dxdydz', 'curv', 'normals', 'fpfh', 'jump_indication', 'edge_meshcnn'
  params.reverse_walk = True
  params.train_min_max_faces2use = [0, 8000]
  params.test_min_max_faces2use = [0, 1000]

  params.net = 'RnnWalkNet'
  params.last_layer_actication = 'softmax'
  params.use_norm_layer = 'InstanceNorm'   #'InstanceNorm' # BatchNorm / InstanceNorm / None
  params.layer_sizes = None

  params.initializers = 'orthogonal'
  params.adjust_vertical_model = False
  params.net_start_from_prev_net = None

  params.net_gru_dropout = 0.5
  params.uniform_starting_point = False
  params.train_max_size_per_class = None    # None / 'uniform_as_max_class' / <a number>

  params.full_accuracy_test = None

  return params

# Semantic Segmentation
# ---------------------
def dancer_params():
  run_name = 'dancer'
  cont_run_number = 0
  params = set_up_default_params('semantic_segmentation', run_name, cont_run_number)

  # Dataset params
  p = '/home/alonlahav/runs/cache/dancer_with_meshcnn_ftrs/'
  params.datasets2use['train'] = [p + '*train*.npz']
  params.datasets2use['test']  = [p + '*test*.npz']
  params.normalize_model = True # -> change to False
  params.sub_mean_for_data_augmentation = True
  #params.train_data_augmentation = {'stride': {'prob2augment': 0.5, 'max_skip_prob': 0.5}, 'scaling': (0.5, 2), 'rotation': 10}

  # Other params
  params.walk_alg = 'no_jumps'   # no_repeat / no_jumps / fast / fastest / only_jumps / local_jumps / no_local_jumps
  params.net_input = ['dxdydz'] # 'xyz', 'dxdydz', 'curv', 'normals', 'fpfh', 'jump_indication', 'edge_meshcnn'
  params.train_min_max_faces2use = [0, 4000]
  params.test_min_max_faces2use = [0, 1000]

  params.n_classes = 16

  params.net_start_from_prev_net = None

  return params

def faust_params():
  params = set_up_default_params('semantic_segmentation', 'faust', 0)

  p = '/home/alonlahav/runs/cache/dancer_full_tmp_label_fixed/'
  params.datasets2use['train'] = ['/home/alonlahav/runs/cache/faust_full_reduced_to_2.4.8k_faces_with_walks/*.npz']
  params.datasets2use['test']  = [p + '*test*.npz']

  #params.n_models_per_test_epoch = 10

  return params

def human_seg_params():

  if 1:
    # |V| = 750 , |F| = 1500 => seq_len = |V| / 2.5 = 300
    params = set_up_default_params('semantic_segmentation', 'human_seg__16models', 430)
    sub_dir = 'human_seg_from_mcnn_with_nrmls'
    params.seq_len = 300
    params.train_min_max_faces2use = [0000, 4000]
    params.test_min_max_faces2use = [0000, 4000]
  else:
    # |V| = 4000 , |F| = 8000 => seq_len = |V| / 2.5 = 1600 -> 400
    params = set_up_default_params('semantic_segmentation', 'human_seg__4k_400steps', 0)
    sub_dir = 'human_seg_4k'
    min_max_faces = 4000
    params.train_min_max_faces2use = [min_max_faces, min_max_faces]
    params.test_min_max_faces2use = [min_max_faces, min_max_faces]
    #params.seq_len = int(min_max_faces / 10)
    params.seq_len = 400

  params.n_classes = 9

  p = os.path.expanduser('~') + '/runs/datasets_processed/' + sub_dir + '/'
  p_subset = os.path.expanduser('~') + '/runs/datasets_processed/human_seg_from_mcnn_with_nrmls-train_subset/2_models/'
  params.datasets2use['train'] = [p + '*train*.npz']
  params.datasets2use['test']  = [p + '*test*.npz']

  params.normalize_model = 0
  params.adjust_vertical_model = 1
  params.train_data_augmentation = {'rotation': 45}
  params.test_data_augmentation = {}
  params.net_input = ['dxdydz']  # 'xyz', 'dxdydz', 'curv', 'normals', 'fpfh', 'jump_indication', 'edge_meshcnn'
  params.walk_alg = 'no_jumps'   # no_repeat / no_jumps / fast / fastest / only_jumps / local_jumps / no_local_jumps

  params.min_seq_len = int(params.seq_len / 2)

  params.full_accuracy_test = {'from_cach_dataset': sub_dir + '/*test*.npz',
                               'n_iters': 32}

  return params

# Semi / Un supervised learning
# -----------------------------
def transfer_learning_params():
  if 0:
    params = set_up_default_params('semantic_segmentation', 'TrnsLrn__HumanSeg_2meshes4trn__BaseLine_NoTrnsLrn', 0)
    params.n_classes = 9
  else:
    params = set_up_default_params('semantic_segmentation', 'TrnsLrn__HumanSeg_2meshes4trn__StartingFromShrec', 0)
    params.net_start_from_prev_net = os.path.expanduser('~') + \
                                     '/runs/mesh_learning/rnn_mesh_walk/0318-03.03.2020..19.29__Shrec16_TestDatasetReorder/learned_model2keep--720000'
    params.n_classes = 30

  params.seq_len = 100
  params.train_min_max_faces2use = [0000, 4000]
  params.test_min_max_faces2use = [0000, 4000]

  params.datasets2use['train'] = [os.path.expanduser('~') +
                                  '/runs/datasets_processed/human_seg_from_mcnn_with_nrmls-train_subset/2_models/*train*.npz']
  params.datasets2use['test']  = [os.path.expanduser('~') +
                                  '/runs/datasets_processed/human_seg_from_mcnn_with_nrmls/*test*.npz']

  params.normalize_model = 0
  params.adjust_vertical_model = 1
  params.train_data_augmentation = {'horisontal_90deg': [0, 2], 'rotation': 45}
  params.test_data_augmentation = {}
  params.net_input = ['dxdydz']  # 'xyz', 'dxdydz', 'curv', 'normals', 'fpfh', 'jump_indication', 'edge_meshcnn'
  params.walk_alg = 'no_jumps'   # no_repeat / no_jumps / fast / fastest / only_jumps / local_jumps / no_local_jumps

  params.min_seq_len = int(params.seq_len / 2)

  params.full_accuracy_test = {'from_cach_dataset': 'human_seg_from_mcnn_with_nrmls/*test*.npz',
                               'n_iters': 32}

  return params


def semi_supervised_human_seg_params():
  params = set_up_default_params('semisupervised', 'human_seg_semi--self_supervised', 0)
  #params = set_up_default_params('semisupervised', 'human_seg_semi--OneLabeledNoUnsup', 0)

  #params.train_dataset_size_limit = 32
  params.n_classes = 9
  params.batch_size = 1

  p = os.path.expanduser('~') + '/runs/datasets_processed/human_seg_with_geodist/'
  if 1:
    params.datasets2use['train'] = [p + '*train_faust__tr_reg_070_not_changed_1500.npz',
                                    p + '*train*.npz']
    params.train_loss = ['cros_entr', 'triplet']
    params.network_tasks = ['semantic_segmentation', 'self:triplets']
  elif 1:
    params.datasets2use['train'] = [p + '*train_faust__tr_reg_070_not_changed_1500.npz',
                                    p + '*train*.npz']
    params.train_loss = ['cros_entr', 'cros_entr']
    params.network_tasks = ['semantic_segmentation', 'self:location_prediction']
  else:
    params.datasets2use['train'] = [p + '*train_faust__tr_reg_070_not_changed_1500.npz']
  params.datasets2use['test']  = [p + '*test*.npz']
  params.train_min_max_faces2use = [0, 4000]
  params.test_min_max_faces2use = [0, 4000]

  params.normalize_model = 0
  params.adjust_vertical_model = 1
  #params.train_data_augmentation = {'horisontal_90deg': [0, 2], 'rotation': 90}
  #params.test_data_augmentation = {}
  params.net_input = ['dxdydz']  # 'xyz', 'dxdydz', 'curv', 'normals', 'fpfh', 'jump_indication', 'edge_meshcnn'
  params.walk_alg = 'no_jumps'   # no_repeat / no_jumps / fast / fastest / only_jumps / local_jumps / no_local_jumps
  params.net_start_from_prev_net = None # '/home/alonlahav/runs/mesh_learning/rnn_mesh_walk/0205-02.02.2020..10.12__human_seg_DxdydzJump_TrnHoriFlip_Rot_Augm/learned_model__350009.keras'

  params.seq_len = 50
  params.min_seq_len = int(params.seq_len * 0.9)

  params.learning_rate = 1e-5

  return params


def unsupervised_human_seg_params():
  params = set_up_default_params('self:triplets', 'human_seg_unsupervised-far_neg_examples', 0)

  #params.train_dataset_size_limit = 64
  params.n_classes = 9
  params.batch_size = 1

  p = os.path.expanduser('~') + '/runs/datasets_processed/human_seg_with_geodist/'
  params.datasets2use['train'] = [p + '*train*.npz']
  params.datasets2use['test']  = None # [p + '*test*.npz']
  params.train_min_max_faces2use = [0, 4000]
  params.test_min_max_faces2use = [0, 4000]

  params.normalize_model = 0
  params.adjust_vertical_model = 1
  #params.train_data_augmentation = {'horisontal_90deg': [0, 2], 'rotation': 90}
  #params.test_data_augmentation = {}
  params.net_input = ['dxdydz']  # 'xyz', 'dxdydz', 'curv', 'normals', 'fpfh', 'jump_indication', 'edge_meshcnn'
  params.walk_alg = 'no_jumps'   # no_repeat / no_jumps / fast / fastest / only_jumps / local_jumps / no_local_jumps
  params.net_start_from_prev_net = None # '/home/alonlahav/runs/mesh_learning/rnn_mesh_walk/0205-02.02.2020..10.12__human_seg_DxdydzJump_TrnHoriFlip_Rot_Augm/learned_model__350009.keras'

  params.seq_len = 50
  params.min_seq_len = int(params.seq_len * 0.9)

  return params

def coseg_params(type): # aliens / chairs / vases
  # |V| = 750 , |F| = 1500 => seq_len = |V| / 2.5 = 300
  sub_folder = 'coseg_' + type
  p = os.path.expanduser('~') + '/runs/datasets_processed/coseg/' + sub_folder + '/'
  if 0:
    params = set_up_default_params('semantic_segmentation', 'coseg_' + type + '_Full_WalkCngLonger', 0)
    params.datasets2use['train'] = [p + '*train*.npz']
  else:
    params = set_up_default_params('semantic_segmentation', 'coseg_' + type + '_1_model4train_WalkCngLonger', 0)
    params.datasets2use['train'] = [p + '/train_subset/001_model_for_train/*train*.npz']

  params.reverse_walk = False

  params.n_classes = 10

  params.datasets2use['test']  = [p + '*test*.npz']

  params.train_min_max_faces2use = [0, 4000]
  params.test_min_max_faces2use = [0, 4000]

  params.normalize_model = 0
  params.adjust_vertical_model = 0
  params.train_data_augmentation = {'rotation': 45}
  params.test_data_augmentation = {}
  params.net_input = ['dxdydz']  # 'xyz', 'dxdydz', 'curv', 'normals', 'fpfh', 'jump_indication', 'edge_meshcnn'
  params.walk_alg = 'no_jumps'   # no_repeat / no_jumps / fast / fastest / only_jumps / local_jumps / no_local_jumps
  params.net_start_from_prev_net = None

  params.seq_len = 300
  params.min_seq_len = int(params.seq_len / 2)

  params.full_accuracy_test = {'from_cach_dataset': 'coseg/' + sub_folder + '/*test*.npz',
                               'n_iters': 32}


  params.learning_rate =       [2e-4, 1e-4,  5e-5,  2e-5 ]
  params.learning_rate_steps = [0,    50e3,  150e3, 300e3, np.inf]

  return params

# Classifications
# ---------------
def modelnet_params():
  params = set_up_default_params('classification', 'modelnet__Adam_pooling', 0)
  params.normalize_model = 1
  p = 'modelnet40_1k2k4k'
  params.datasets2use['train'] = [os.path.expanduser('~') + '/runs/datasets_processed/' + p + '/*train*.npz']
  params.datasets2use['test'] = [os.path.expanduser('~') + '/runs/datasets_processed/' + p + '/*test*.npz']
  #params.train_data_augmentation = {'aspect_ratio': 0.5}
  params.train_min_max_faces2use = [0, 4000]
  params.test_min_max_faces2use = [0, 1000]
  params.reverse_walk = True
  if 0: # best so far
    params.net_input = ['xyz', 'jump_indication']                # 'xyz', 'dxdydz', 'curv', 'normals', 'fpfh', 'jump_indication'
    params.walk_alg = 'no_repeat'   # no_repeat / no_jumps / fast / fastest / only_jumps / local_jumps / no_local_jumps
  elif 0:
    params.net_input = ['xyz', 'jump_indication']                # 'xyz', 'dxdydz', 'curv', 'normals', 'fpfh', 'jump_indication'
    params.walk_alg = 'no_jumps'   # no_repeat / no_jumps / fast / fastest / only_jumps / local_jumps / no_local_jumps
  elif 1:
    if 1:
      params.optimizer_type = 'adam'  # sgd / adam / cycle
      params.learning_rate_dynamics = 'adam'
      params.cycle_opt_prms = EasyDict({'initial_learning_rate': 5e-6,
                                        'maximal_learning_rate': 2e-4,
                                        'step_size': 10000})

    params.seq_len = 200
    params.min_seq_len = int(params.seq_len / 2)
    params.train_min_max_faces2use = [0, 1000]
    params.reverse_walk = False
    params.net_input = ['xyz']  # 'xyz', 'dxdydz', 'curv', 'normals', 'fpfh', 'jump_indication'
    params.walk_alg = 'no_repeats'  # no_repeat / no_jumps / fast / fastest / only_jumps / local_jumps / no_local_jumps
    #params.layer_sizes = {'fc1': 128, 'fc2': 256, 'gru1': 256, 'gru2': 256, 'gru3': 256}
    params.aditional_network_params = ['pooling']
  else:
    params.net_input = ['xyz']  # 'xyz', 'dxdydz', 'curv', 'normals', 'fpfh', 'jump_indication'
    params.walk_alg = 'local_jumps'  # no_repeat / no_jumps / fast / fastest / only_jumps / local_jumps / no_local_jumps
  params.n_classes = 40

  params.train_max_size_per_class = None    # None / 'uniform_as_max_class' / <a number>

  params.full_accuracy_test = {'from_cach_dataset': 'modelnet40_1k2k4k/*test*.npz',
                               'labels': dataset_prepare.model_net_labels,
                               'min_max_faces2use': [0, 1000],
                               }

  return params

def cubes_params():
  # |V| = 250 , |F| = 500 => seq_len = |V| / 2.5 = 100
  params = set_up_default_params('classification', 'Cubes_', 16)

  params.reverse_walk = False # To be checked!

  params.seq_len = 100
  params.min_seq_len = int(params.seq_len / 2)

  p = 'cubes'
  params.datasets2use['train'] = [os.path.expanduser('~') + '/runs/datasets_processed/' + p + '/*train*.npz']
  params.datasets2use['test'] = [os.path.expanduser('~') + '/runs/datasets_processed/' + p + '/*test*.npz']
  params.network_task = 'classification'
  params.train_data_augmentation = {}
  params.train_min_max_faces2use = [0, np.inf]
  params.test_min_max_faces2use = [0, np.inf]

  params.net_input = ['dxdydz']                # 'xyz', 'dxdydz', 'curv', 'normals', 'fpfh', 'jump_indication'
  params.walk_alg = 'no_jumps'   # no_repeat / no_jumps / fast / fastest / only_jumps / local_jumps / no_local_jumps

  params.n_classes = 22

  params.full_accuracy_test = {'from_cach_dataset': 'cubes/*test*.npz',
                               'labels': dataset_prepare.cubes_labels,
                               }

  return params

def shrec11_params():
  # |V| = 250 , |F| = 500 => seq_len = |V| / 2.5 = 100
  params = set_up_default_params('classification', 'Shrec11_16-04C', 28)

  params.reverse_walk = False

  params.seq_len = 100
  params.normalize_model = 1

  if 0:
    p = 'shrec11'
    params.datasets2use['train'] = [os.path.expanduser('~') + '/runs/datasets_processed/' + p + '/*train*.npz']
    params.datasets2use['test'] = [os.path.expanduser('~') + '/runs/datasets_processed/' + p + '/*test*.npz']
  elif 1:
    subdiv = 'shrec11/16-04_C/'
    params.datasets2use['train'] = [os.path.expanduser('~') + '/runs/datasets_processed/' + subdiv + 'train/*.npz']
    params.datasets2use['test']  = [os.path.expanduser('~') + '/runs/datasets_processed/' + subdiv + 'test/*.npz']
  else:
    params.seq_len = 400  # should be 800 steps long
    subdiv = 'shrec11_raw_4k/11-04_A/'
    params.datasets2use['train'] = [os.path.expanduser('~') + '/runs/datasets_processed/' + subdiv + 'train/*.npz']
    params.datasets2use['test']  = [os.path.expanduser('~') + '/runs/datasets_processed/' + subdiv + 'test/*.npz']

  params.min_seq_len = int(params.seq_len / 2)

  params.train_data_augmentation = {'rotation': 45}
  params.train_min_max_faces2use = [0, np.inf]
  params.test_min_max_faces2use = [0, np.inf]

  if 0: # best configuration (to be confirmed)
    params.net_input = ['xyz', 'jump_indication']  # 'xyz', 'dxdydz', 'curv', 'normals', 'fpfh', 'jump_indication'
    params.walk_alg = 'no_jumps'  # no_repeat / no_jumps / fast / fastest / only_jumps / local_jumps / no_local_jumps
  else:
    params.net_input = ['dxdydz']                # 'xyz', 'dxdydz', 'curv', 'normals', 'fpfh', 'jump_indication'
    params.walk_alg = 'no_jumps'   # no_repeat / no_jumps / fast / fastest / only_jumps / local_jumps / no_local_jumps

  params.n_classes = 30

  params.full_accuracy_test = {'from_cach_dataset': subdiv + 'test/*.npz',
                               'labels': dataset_prepare.shrec11_labels}

  return params

# =============== Ran's addition - pytorch compatible ============ #
def modelnet_params():
  params = set_up_default_params('classification', 'modelnet__Pooling_Adam', 0)
  params.normalize_model = 1
  p = 'modelnet40_1k2k4k'
  params.datasets2use['train'] = ['./Data/' + p + '/*train*.npz']
  params.datasets2use['test'] = ['./Data/' + p + '/*test*.npz']
  #params.train_data_augmentation = {'aspect_ratio': 0.5}
  params.train_min_max_faces2use = [0, 4000]
  params.test_min_max_faces2use = [0, 1000]
  params.reverse_walk = False
  if 0: # best so far
    params.net_input = ['xyz', 'jump_indication']                # 'xyz', 'dxdydz', 'curv', 'normals', 'fpfh', 'jump_indication'
    params.walk_alg = 'no_repeat'   # no_repeat / no_jumps / fast / fastest / only_jumps / local_jumps / no_local_jumps
  elif 0:
    params.net_input = ['xyz', 'jump_indication']                # 'xyz', 'dxdydz', 'curv', 'normals', 'fpfh', 'jump_indication'
    params.walk_alg = 'no_jumps'   # no_repeat / no_jumps / fast / fastest / only_jumps / local_jumps / no_local_jumps
  elif 1:
    if 1:
      params.optimizer_type = 'adam'  # sgd / adam / cycle
      params.learning_rate_dynamics = 'cycle'
      params.cycle_opt_prms = EasyDict({'initial_learning_rate': 5e-6,
                                        'maximal_learning_rate': 2e-4,
                                        'step_size': 10000})

    params.seq_len = 200
    params.min_seq_len = int(params.seq_len / 2)
    params.train_min_max_faces2use = [0, 1000]
    params.net_input = ['xyz']  # 'xyz', 'dxdydz', 'curv', 'normals', 'fpfh', 'jump_indication'
    params.walk_alg = 'no_jumps'  # no_repeat / no_jumps / fast / fastest / only_jumps / local_jumps / no_local_jumps
    #params.layer_sizes = {'fc1': 128, 'fc2': 256, 'gru1': 256, 'gru2': 256, 'gru3': 256}
    params.aditional_network_params = ['pooling']
  else:
    params.net_input = ['xyz']  # 'xyz', 'dxdydz', 'curv', 'normals', 'fpfh', 'jump_indication'
    params.walk_alg = 'local_jumps'  # no_repeat / no_jumps / fast / fastest / only_jumps / local_jumps / no_local_jumps
  params.n_classes = 40

  params.train_max_size_per_class = None    # None / 'uniform_as_max_class' / <a number>

  params.full_accuracy_test = {'from_cach_dataset': 'modelnet40_1k2k4k/*test*.npz',
                               'labels': dataset_prepare.model_net_labels,
                               'min_max_faces2use': [0, 1000],
                               }

  # Taken from setup_features_params
  if params.uniform_starting_point:
    params.area = 'all'
  else:
    params.area = -1
  params.set_seq_len_by_n_faces = False
  params.support_mesh_cnn_ftrs = False
  params.fill_features_functions = []
  params.number_of_features = 0
  if 'dxdydz' in params.net_input:
    params.fill_features_functions.append(mesh.fill_dxdydz_features)
    params.number_of_features += 3
  elif 'xyz' in params.net_input:
    params.fill_features_functions.append(mesh.fill_xyz_features)
    params.number_of_features += 3
  params.edges_needed = True
  if params.walk_alg == 'no_repeat':
    params.walk_function = walks.get_seq_random_walk_no_repeat
    params.kdtree_query_needed = True
  elif params.walk_alg == 'no_jumps':
    params.walk_function = walks.get_seq_random_walk_no_jumps
    params.kdtree_query_needed = False

  if params.network_task == 'classification':
    params.label_per_step = False
  elif params.network_task == 'semantic_segmentation':
    params.label_per_step = True
  elif params.network_task == 'self:triplets':
    params.label_per_step = True
  return params


def shapenet_params(pretrained=None):
  params = set_up_default_params('classification', 'shapenet_modelnet_features', 0)
  params.normalize_model = 1
  p = 'modelnet40_1k2k4k'
  params.datasets2use['train'] = [os.path.expanduser('~') + '/PycharmProjects/MeshWalker_Pytorch/Data/shapenet_tmp/' + p + '/*train*.npz']
  params.datasets2use['test'] = [os.path.expanduser('~') + '/runs/datasets_processed/' + p + '/*test*.npz']
  #params.train_data_augmentation = {'aspect_ratio': 0.5}
  params.train_min_max_faces2use = [0, 1000]
  params.test_min_max_faces2use = [0, 1000]
  if 0: # best so far
    params.net_input = ['xyz', 'jump_indication']                # 'xyz', 'dxdydz', 'curv', 'normals', 'fpfh', 'jump_indication'
    params.walk_alg = 'no_repeat'   # no_repeat / no_jumps / fast / fastest / only_jumps / local_jumps / no_local_jumps
  elif 0:
    params.net_input = ['xyz', 'jump_indication']                # 'xyz', 'dxdydz', 'curv', 'normals', 'fpfh', 'jump_indication'
    params.walk_alg = 'no_jumps'   # no_repeat / no_jumps / fast / fastest / only_jumps / local_jumps / no_local_jumps
  elif 1:
    if 1:
      params.optimizer_type = 'cycle'  # sgd / adam / cycle
      params.learning_rate_dynamics = 'cycle'
      params.cycle_opt_prms = EasyDict({'initial_learning_rate': 5e-6,
                                        'maximal_learning_rate': 2e-4,
                                        'step_size': 10000})

    params.seq_len = 400
    params.min_seq_len = int(params.seq_len / 2)
    params.train_min_max_faces2use = [0, 1000]
    params.reverse_walk = False
    params.net_input = ['xyz']  # 'xyz', 'dxdydz', 'curv', 'normals', 'fpfh', 'jump_indication'
    params.walk_alg = 'no_repeat'  # no_repeat / no_jumps / fast / fastest / only_jumps / local_jumps / no_local_jumps
    #params.layer_sizes = {'fc1': 128, 'fc2': 256, 'gru1': 256, 'gru2': 256, 'gru3': 256}
    params.aditional_network_params = ['pooling']
  else:
    params.net_input = ['xyz']  # 'xyz', 'dxdydz', 'curv', 'normals', 'fpfh', 'jump_indication'
    params.walk_alg = 'local_jumps'  # no_repeat / no_jumps / fast / fastest / only_jumps / local_jumps / no_local_jumps
  params.n_classes = 40
  params.train_max_size_per_class = None    # None / 'uniform_as_max_class' / <a number>
  params.full_accuracy_test = {'from_cach_dataset': 'modelnet40_1k2k4k/*test*.npz',
                               'labels': dataset_prepare.model_net_labels,
                               'min_max_faces2use': [0, 1000],
                               }

  # Taken from setup_features_params
  if params.uniform_starting_point:
    params.area = 'all'
  else:
    params.area = -1
  params.set_seq_len_by_n_faces = False
  params.support_mesh_cnn_ftrs = False
  params.fill_features_functions = []
  params.number_of_features = 0
  if 'dxdydz' in params.net_input:
    params.fill_features_functions.append(mesh.fill_dxdydz_features)
    params.number_of_features += 3
  params.edges_needed = True
  if params.walk_alg == 'no_repeat':
    params.walk_function = walks.get_seq_random_walk_no_repeat
    params.kdtree_query_needed = True
  params.pretrained = pretrained
  return params
