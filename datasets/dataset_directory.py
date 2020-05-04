import time, glob, os, shutil, copy

import trimesh
import networkx as nx
from pylab import plt
from easydict import EasyDict
import numpy as np

from utils import utils, walks
from datasets import dataset_prepare

# TODOs:
# - finish walk for model area : add to npz file (realtime calculation increase epoch time from 62 to 65 sec)
# - Add more features
# - Make it faster
# - Once in a while, recalculate the walk seq (?)

# Glabal list of dataset parameters
dataset_params_list = []

def load_model_from_npz(npz_fn):
  if npz_fn.find(':') != -1:
    npz_fn = npz_fn.split(':')[1]
  mesh_data = np.load(npz_fn, encoding='latin1', allow_pickle=True)
  return mesh_data

def get_label_selfsupervised(v, mode='height'):
  if mode == 'hight':
    h = v[0]
    if h >= 1.2:
      return 0
    if h < 1.2 and h >= 1.0:
      return 1
    if h < 1.0 and h >= 0.8:
      return 2
    if h < 0.8 and h >= 0.6:
      return 3
    if h < 0.6:
      return 4

def prepare_area_vertices_list(vertices):
  area_vertices_list = {}

  # Center:
  center = np.linalg.norm(vertices, axis=1) < 0.4
  area_vertices_list[0] = np.where(center)[0]
  next_idx = 1
  pos_x = vertices[:, 0] > 0
  pos_y = vertices[:, 1] > 0
  pos_z = vertices[:, 2] > 0
  for x in [-1, 1]:
    condx = pos_x if x == 1 else np.logical_not(pos_x)
    for y in [-1, 1]:
      condy = pos_y if y == 1 else np.logical_not(pos_y)
      for z in [-1, 1]:
        condz = pos_z if z == 1 else np.logical_not(pos_z)
        area_vertices_list[next_idx] = np.where(condx * condy * condz * np.logical_not(center))[0]
        next_idx += 1

  return area_vertices_list

def norm_model(vertices):
  # Move the model so the bbox center will be at (0, 0, 0)
  mean = np.mean((np.min(vertices, axis=0), np.max(vertices, axis=0)), axis=0)
  vertices -= mean

  # Scale model to fit into the unit ball
  abs_max = np.max(vertices)
  vertices /= abs_max

  if norm_model.sub_mean_for_data_augmentation:
    vertices -= np.nanmean(vertices, axis=0)

def data_augmentation_axes_rot(vertices):
  if np.random.randint(2):    # 50% chance to switch the two hirisontal axes
    vertices[:] = vertices[:, data_augmentation_axes_rot.flip_axes]
  if np.random.randint(2):    # 50% chance to neg one random hirisontal axis
    i = np.random.choice(data_augmentation_axes_rot.hori_axes)
    vertices[:, i] = -vertices[:, i]

def data_augmentation_rotation(vertices):
  if np.random.randint(2):    # 50% chance
    max_rot_ang_deg = data_augmentation_rotation.max_rot_ang_deg
    x = np.random.uniform(-max_rot_ang_deg, max_rot_ang_deg) * np.pi / 180
    y = np.random.uniform(-max_rot_ang_deg, max_rot_ang_deg) * np.pi / 180
    z = np.random.uniform(-max_rot_ang_deg, max_rot_ang_deg) * np.pi / 180
    A = np.array(((np.cos(x), -np.sin(x), 0),
                  (np.sin(x), np.cos(x), 0),
                  (0, 0, 1)),
                 dtype=vertices.dtype)
    B = np.array(((np.cos(y), 0, -np.sin(y)),
                  (0, 1, 0),
                  (np.sin(y), 0, np.cos(y))),
                 dtype=vertices.dtype)
    C = np.array(((1, 0, 0),
                  (0, np.cos(z), -np.sin(z)),
                  (0, np.sin(z), np.cos(z))),
                 dtype=vertices.dtype)
    np.dot(vertices, A, out=vertices)
    np.dot(vertices, B, out=vertices)
    np.dot(vertices, C, out=vertices)

def data_augmentation_aspect_ratio(vertices):
  if np.random.randint(2):    # 50% chance
    for i in range(3):
      r = np.random.uniform(1 - data_augmentation_aspect_ratio.max_ratio, 1 + data_augmentation_aspect_ratio.max_ratio)
      vertices[i] *= r

def fill_xyz_features(features, f_idx, vertices, mesh_extra, seq, jumps, seq_len):
  walk = vertices[seq[1:seq_len + 1]]
  features[:, f_idx:f_idx + walk.shape[1]] = walk
  f_idx += 3
  return f_idx

def fill_dxdydz_features(features, f_idx, vertices, mesh_extra, seq, jumps, seq_len):
  walk = np.diff(vertices[seq[:seq_len + 1]], axis=0) * 100
  features[:, f_idx:f_idx + walk.shape[1]] = walk
  f_idx += 3
  return f_idx

def fill_edge_meshcnn_features(features, f_idx, vertices, mesh_extra, seq, jumps, seq_len):
  edges_map = mesh_extra['edges_map']
  edge_features = mesh_extra['edge_features']
  e_idxs = []
  for i in range(seq_len):
    e1 = seq[i]
    e2 = seq[i + 1]
    e_idxs.append(edges_map[e1, e2])
  walk = edge_features[e_idxs]
  features[:, f_idx:f_idx + walk.shape[1]] = walk
  f_idx += walk.shape[1]
  return f_idx

def fill_vertex_indices(features, f_idx, vertices, mesh_extra, seq, jumps, seq_len):
  walk = seq[1:seq_len + 1][:, None]
  features[:, f_idx:f_idx + walk.shape[1]] = walk
  f_idx += 1
  return f_idx

def fill_jumps(features, f_idx, vertices, mesh_extra, seq, jumps, seq_len):
  walk = jumps[1:seq_len + 1][:, None]
  features[:, f_idx:f_idx + walk.shape[1]] = walk
  f_idx += 1
  return f_idx

def setup_data_augmentation(dataset_params, data_augmentation):
  dataset_params.data_augmentaion_vertices_functions = []
  if 'horisontal_90deg' in data_augmentation.keys() and data_augmentation['horisontal_90deg']:
    dataset_params.data_augmentaion_vertices_functions.append(data_augmentation_axes_rot)
    data_augmentation_axes_rot.hori_axes = data_augmentation['horisontal_90deg']
    flip_axes_ = [0, 1, 2]
    data_augmentation_axes_rot.flip_axes  = [0, 1, 2]
    data_augmentation_axes_rot.flip_axes[data_augmentation_axes_rot.hori_axes[0]] = flip_axes_[data_augmentation_axes_rot.hori_axes[1]]
    data_augmentation_axes_rot.flip_axes[data_augmentation_axes_rot.hori_axes[1]] = flip_axes_[data_augmentation_axes_rot.hori_axes[0]]
  if 'rotation' in data_augmentation.keys() and data_augmentation['rotation']:
    data_augmentation_rotation.max_rot_ang_deg = data_augmentation['rotation']
    dataset_params.data_augmentaion_vertices_functions.append(data_augmentation_rotation)
  if 'aspect_ratio' in data_augmentation.keys() and data_augmentation['aspect_ratio']:
    data_augmentation_aspect_ratio.max_ratio = data_augmentation['aspect_ratio']
    dataset_params.data_augmentaion_vertices_functions.append(data_augmentation_aspect_ratio)

def setup_features_params(dataset_params, params):
  if params.uniform_starting_point:
    dataset_params.area = 'all'
  else:
    dataset_params.area = -1
  norm_model.sub_mean_for_data_augmentation = params.sub_mean_for_data_augmentation
  dataset_params.support_mesh_cnn_ftrs = False
  dataset_params.fill_features_functions = []
  dataset_params.number_of_features = 0
  net_input = params.net_input
  if 'xyz' in net_input:
    dataset_params.fill_features_functions.append(fill_xyz_features)
    dataset_params.number_of_features += 3
  if 'dxdydz' in net_input:
    dataset_params.fill_features_functions.append(fill_dxdydz_features)
    dataset_params.number_of_features += 3
  if 'edge_meshcnn' in net_input:
    dataset_params.support_mesh_cnn_ftrs = True
    dataset_params.fill_features_functions.append(fill_edge_meshcnn_features)
    dataset_params.number_of_features += 5
  if 'jump_indication' in net_input:
    dataset_params.fill_features_functions.append(fill_jumps)
    dataset_params.number_of_features += 1
  if 'vertex_indices' in net_input:
    dataset_params.fill_features_functions.append(fill_vertex_indices)
    dataset_params.number_of_features += 1

  dataset_params.edges_needed = True
  if params.walk_alg == 'no_repeat':
    dataset_params.walk_function = walks.get_seq_random_walk_no_repeat
    dataset_params.kdtree_query_needed = True
  elif params.walk_alg == 'no_jumps':
    dataset_params.walk_function = walks.get_seq_random_walk_no_jumps
    dataset_params.kdtree_query_needed = False
  elif params.walk_alg == 'no_local_jumps':
    dataset_params.walk_function = walks.get_seq_random_walk_no_local_jumps
    dataset_params.kdtree_query_needed = False
  elif params.walk_alg == 'fast':
    dataset_params.walk_function = walks.get_seq_random_walk_fast
    dataset_params.kdtree_query_needed = True
  elif params.walk_alg == 'fastest':
    dataset_params.walk_function = walks.get_seq_random_walk_fastest
    dataset_params.kdtree_query_needed = False
  elif params.walk_alg == 'local_jumps':
    dataset_params.walk_function = walks.get_seq_random_walk_local_jumps
    dataset_params.kdtree_query_needed = True
    dataset_params.edges_needed = False
  elif params.walk_alg == 'only_jumps':
    dataset_params.walk_function = walks.get_seq_random_walk_fastest_only_jumps
    dataset_params.kdtree_query_needed = False
    dataset_params.edges_needed = False
  else:
    raise Exception('Walk alg not recognized: ' + params.walk_alg)

  return dataset_params.number_of_features

def get_starting_point(area, area_vertices_list, n_vertices, walk_id):
  if area is None or area_vertices_list is None:
    return np.random.randint(n_vertices)
  elif area == -1:
    candidates = np.zeros((0,))
    while candidates.size == 0:
      b = np.random.randint(9)
      candidates = area_vertices_list[b]
    return np.random.choice(candidates)
  else:
    candidates = area_vertices_list[walk_id % len(area_vertices_list)]
    while candidates.size == 0:
      b = np.random.randint(9)
      candidates = area_vertices_list[b]
    return np.random.choice(candidates)

def generate_walk_py_fun(fn, vertices, faces, edges, kdtree_query, labels, params_idx):
  return tf.py_function(
    generate_walk,
    inp=(fn, vertices, faces, edges, kdtree_query, labels, params_idx),
    Tout=(fn.dtype, vertices.dtype, tf.int32)
  )


def generate_walk(fn, vertices, faces, edges, kdtree_query, labels_from_npz, params_idx):
  mesh_data = {'vertices': vertices.numpy(),
               'faces': faces.numpy(),
               'edges': edges.numpy(),
               'kdtree_query': kdtree_query.numpy(),
               }
  if dataset_params_list[params_idx[0]].label_per_step:
    mesh_data['labels'] = labels_from_npz.numpy()

  dataset_params = dataset_params_list[params_idx[0].numpy()]
  features, labels = mesh_data_to_walk_features(mesh_data, dataset_params)

  if dataset_params_list[params_idx[0]].label_per_step:
    labels_return = labels
  else:
    labels_return = labels_from_npz

  return fn[0], features, labels_return

def mesh_data_to_walk_features(mesh_data, dataset_params):
  vertices = mesh_data['vertices']
  seq_len = dataset_params.seq_len
  if dataset_params.set_seq_len_by_n_faces:
    if mesh_data['faces'].shape[0] > 1000:
      seq_len = int(mesh_data['faces'].shape[0] / 1000 * 100)

  # Preprocessing
  if dataset_params.adjust_vertical_model:
    vertices[:, 1] -= vertices[:, 1].min()
  if dataset_params.normalize_model:
    norm_model(vertices)

  # Data augmentation
  for data_augmentaion_function in dataset_params.data_augmentaion_vertices_functions:
    data_augmentaion_function(vertices)

  # Get essential data from file
  if dataset_params.label_per_step:
    mesh_labels = mesh_data['labels']
  else:
    mesh_labels = -1 * np.ones((vertices.shape[0],))

  if dataset_params.support_mesh_cnn_ftrs:
    edges_map = mesh_data['edges_map'].tolist()
    mesh_extra = {'edge_features': mesh_data['edge_features'],
                  'edges_map': edges_map}
  else:
    mesh_extra = {}
  mesh_extra['n_vertices'] = vertices.shape[0]
  if dataset_params.edges_needed:
    mesh_extra['edges'] = mesh_data['edges']
  if dataset_params.kdtree_query_needed:
    mesh_extra['kdtree_query'] = mesh_data['kdtree_query']

  features = np.zeros((dataset_params.n_walks_per_model, seq_len, dataset_params.number_of_features), dtype=np.float32)
  labels   = np.zeros((dataset_params.n_walks_per_model, seq_len), dtype=np.int32)

  if 'area_vertices_list' in mesh_data.keys():
    area_vertices_list = mesh_data['area_vertices_list']
    if type(mesh_data['area_vertices_list']) is not dict:
      area_vertices_list = area_vertices_list.tolist()
  else:
    #raise Exception('area_vertices_list not there')
    area_vertices_list = prepare_area_vertices_list(vertices)

  if mesh_data_to_walk_features.SET_SEED_WALK:
    np.random.seed(mesh_data_to_walk_features.SET_SEED_WALK)
  if dataset_params.network_task == 'self:triplets':
    neg_walk_f0 = np.random.randint(vertices.shape[0])
    if 1:
      pos_walk_f0 = np.random.choice(mesh_data['far_vertices'][neg_walk_f0])
    else:
      pos_walk_f0 = np.random.choice(mesh_data['mid_vertices'][neg_walk_f0])
  for walk_id in range(dataset_params.n_walks_per_model):
    if dataset_params.network_task == 'self:triplets':
      if walk_id < dataset_params.n_walks_per_model / 2:
        f0 = neg_walk_f0
      else:
        f0 = pos_walk_f0
    else:
      f0 = get_starting_point(dataset_params.area, area_vertices_list, vertices.shape[0], walk_id)
    if mesh_data_to_walk_features.SET_SEED_WALK:
      f0 = mesh_data_to_walk_features.SET_SEED_WALK

    seq, jumps = dataset_params.walk_function(mesh_extra, f0, seq_len)
    #tf.print(mesh_data['faces'].sum(), 'f0=', f0, seq[:5])
    if dataset_params.reverse_walk:
      seq = seq[::-1]
      jumps = jumps[::-1]

    f_idx = 0
    for fill_ftr_fun in dataset_params.fill_features_functions:
      f_idx = fill_ftr_fun(features[walk_id], f_idx, vertices, mesh_extra, seq, jumps, seq_len)
    if dataset_params.label_per_step:
      if dataset_params.network_task == 'self:triplets':
        labels[walk_id] = seq[1:seq_len + 1]
      else:
        labels[walk_id] = mesh_labels[seq[1:seq_len + 1]]

  return features, labels

def dbg_get_title_to_show_mesh(mesh_data):
  label = mesh_data['label']
  label_name = utils.label2shape_modelnet[label]
  return str(label) + ' , ' + label_name

def load_data_classification(file, params_idx):
  file_str = file
  mesh_data = np.load(file_str, encoding='latin1', allow_pickle=True)
  features, labels = mesh_data_to_walk_features(mesh_data, dataset_params_list[params_idx])
  label = mesh_data['label']
  name = str(mesh_data['dataset_name']) + ':' + file_str
  if dataset_params_list[params_idx].label_per_step:
    return name, features, labels
  else:
    return name, features, label

def get_file_names(pathname_expansion, min_max_faces2use):
  filenames_ = glob.glob(pathname_expansion)
  filenames = []
  for fn in filenames_:
    try:
      n_faces = int(fn.split('.')[-2].split('_')[-1])
      if n_faces > min_max_faces2use[1] or n_faces < min_max_faces2use[0]:
        continue
    except:
      pass
    filenames.append(fn)
  assert len(filenames) > 0, 'DATASET error: no files in directory to be used! \nDataset directory: ' + pathname_expansion

  return filenames


def adjust_fn_list_by_size(filenames_, max_size_per_class):
  lmap = dataset_prepare.map_fns_to_label(filenames=filenames_)
  filenames = []
  if type(max_size_per_class) is int:
    for k, v in lmap.items():
      for i, f in enumerate(v):
        if i >= max_size_per_class:
          break
        filenames.append(f)
  elif max_size_per_class == 'uniform_as_max_class':
    max_size = 0
    for k, v in lmap.items():
      if len(v) > max_size:
        max_size = len(v)
    for k, v in lmap.items():
      f = int(np.ceil(max_size / len(v)))
      fnms = v * f
      filenames += fnms[:max_size]
  else:
    raise Exception('max_size_per_class not recognized')

  return filenames


def filter_fn_by_class(filenames_, classes_indices_to_use):
  filenames = []
  for fn in filenames_:
    mesh_data = np.load(fn, encoding='latin1', allow_pickle=True)
    if classes_indices_to_use is not None and mesh_data['label'] not in classes_indices_to_use:
      continue
    if 0:
      mesh = trimesh.Trimesh(mesh_data['vertices'], mesh_data['faces'])
      n_con_components = nx.number_connected_components(mesh.vertex_adjacency_graph)
      if n_con_components > 1:
        continue
    filenames.append(fn)
  return filenames

def setup_dataset_params(params, data_augmentation):
  p_idx = len(dataset_params_list)
  ds_params = copy.deepcopy(params)
  ds_params.set_seq_len_by_n_faces = False

  setup_data_augmentation(ds_params, data_augmentation)
  setup_features_params(ds_params, params)

  dataset_params_list.append(ds_params)

  return p_idx

# class OpenMeshDataset(tf.data.Dataset):
#   # OUTPUT:      (fn,               vertices,          faces,           edges,           kdtree_query,    labels,          params_idx)
#   OUTPUT_TYPES = (tf.dtypes.string, tf.dtypes.float32, tf.dtypes.int16, tf.dtypes.int16, tf.dtypes.int16, tf.dtypes.int32, tf.dtypes.int16)
#
#   def _generator(fn_, params_idx):
#     fn = fn_[0]
#     with np.load(fn, encoding='latin1', allow_pickle=True) as mesh_data:
#       vertices = mesh_data['vertices']
#       faces = mesh_data['faces']
#       edges = mesh_data['edges']
#       if dataset_params_list[params_idx].label_per_step:
#         labels = mesh_data['labels']
#       else:
#         labels = mesh_data['label']
#       if dataset_params_list[params_idx].kdtree_query_needed:
#         kdtree_query = mesh_data['kdtree_query']
#       else:
#         kdtree_query = [-1]
#
#       name = mesh_data['dataset_name'].tolist() + ':' + fn.decode()
#
#     yield ([name], vertices, faces, edges, kdtree_query, labels, [params_idx])
#
#   def __new__(cls, filenames, params_idx):
#     return tf.data.Dataset.from_generator(
#       cls._generator,
#       output_types=cls.OUTPUT_TYPES,
#       args=(filenames, params_idx)
#     )

def dump_all_fns_to_file(filenames, params):
  if 'logdir' in params.keys():
    for n in range(10):
      log_fn = params.logdir + '/dataset_files_' + str(n).zfill(2) + '.txt'
      if not os.path.isfile(log_fn):
        try:
          with open(log_fn, 'w') as f:
            for fn in filenames:
              f.write(fn + '\n')
        except:
          pass
        break

def check_fns():
  path = '/home/alonlahav/runs/mesh_learning/rnn_mesh_walk/0031-07.04.2020..09.59__Shrec11_16-04A_4kFaces/'
  f1 = path + 'dataset_files_00.txt'
  f2 = path + 'dataset_files_01.txt'
  fns1 = []
  fns2 = []
  for line in open(f1):
    fns1.append(line.split('/')[-1].split('_')[0])
  for line in open(f2):
    fns2.append(line.split('/')[-1].split('_')[0])
  number_of_intersections = 0
  for n in fns1:
    if n in fns2:
      number_of_intersections += 1
  for n in fns2:
    if n in fns1:
      number_of_intersections += 1
  print('Number of files in 1st log:', len(fns1))
  print('Number of files in 2nd log:', len(fns2))
  print('number_of_intersections:', number_of_intersections)

def tf_mesh_dataset(params, pathname_expansion, mode=None, size_limit=np.inf, shuffle_size=1000,
                    permute_file_names=True, min_max_faces2use=[0, np.inf], data_augmentation={},
                    must_run_on_all=False, max_size_per_class=None, min_dataset_size=16):
  LIKE_TF_EXAMPLE = 1

  params_idx = setup_dataset_params(params, data_augmentation)
  number_of_features = dataset_params_list[params_idx].number_of_features
  params.net_input_dim = number_of_features
  mesh_data_to_walk_features.SET_SEED_WALK = 0

  filenames = get_file_names(pathname_expansion, min_max_faces2use)
  if params.classes_indices_to_use is not None:
    filenames = filter_fn_by_class(filenames, params.classes_indices_to_use)
  if max_size_per_class is not None:
    filenames = adjust_fn_list_by_size(filenames, max_size_per_class)

  if permute_file_names:
    filenames = np.random.permutation(filenames)
  else:
    filenames.sort()
  if size_limit < len(filenames):
    filenames = filenames[:size_limit]
  n_items = len(filenames)
  if len(filenames) < min_dataset_size:
    filenames = filenames.tolist() * (int(min_dataset_size / len(filenames)) + 1)

  if mode == 'classification':
    dataset_params_list[params_idx].label_per_step = False
  elif mode == 'semantic_segmentation':
    dataset_params_list[params_idx].label_per_step = True
  elif mode == 'self:triplets':
    dataset_params_list[params_idx].label_per_step = True
  else:
    raise Exception('DS mode ?')

  dump_all_fns_to_file(filenames, params)

  if LIKE_TF_EXAMPLE:
    def _open_npz_fn(*args):
      return OpenMeshDataset(args, params_idx)

    ds = tf.data.Dataset.from_tensor_slices(filenames)
    if shuffle_size:
      ds = ds.shuffle(shuffle_size)
    ds = ds.interleave(_open_npz_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.cache()
    ds = ds.map(generate_walk_py_fun, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(params.batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
  else:
    def _gen_ds_for_inter(filenames):
      ds_ = tf.data.Dataset.from_tensor_slices(filenames)
      ds_ = ds_.map(lambda file: tf.py_function(func=load_data_classification,
                                                inp=[file, params_idx],
                                                Tout=(tf.string, tf.float32, tf.int32)))
      return ds_

    if must_run_on_all:
      n_chunks = 1
    else:
      n_chunks = 8
      len_cut = int(len(filenames) / n_chunks) * n_chunks
      filenames = filenames[:len_cut]
    filenames_chunks = np.array_split(np.array(filenames), n_chunks)
    ds = tf.data.Dataset.from_tensor_slices(filenames_chunks)
    if shuffle_size:
      ds = ds.shuffle(shuffle_size)
    ds = ds.interleave(_gen_ds_for_inter, cycle_length=4, block_length=16, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(params.batch_size, drop_remainder=False)

  return ds, n_items

def mesh_dataset_iterator(params, pathname_expansion, mode=None, size_limit=np.inf, shuffle_size=1000,
                          permute_file_names=True, min_max_faces2use=[0, np.inf], data_augmentation={}):
  '''
  Simulates tf dataset for debug (not including batching)
  '''
  params_idx = setup_dataset_params(params, data_augmentation)
  number_of_features = dataset_params_list[params_idx].number_of_features
  params.net_input_dim = number_of_features
  mesh_data_to_walk_features.SET_SEED_WALK = 0

  filenames = get_file_names(pathname_expansion, min_max_faces2use)
  if params.classes_indices_to_use is not None:
    filenames = filter_fn_by_class(filenames, params.classes_indices_to_use)

  if 0:
    filenames = [f for f in filenames if f.find('shrec_6') != -1 or f.find('shrec_15') != -1]

  filenames = np.random.permutation(filenames)
  if permute_file_names:
    filenames = np.random.permutation(filenames)
  else:
    filenames.sort()
  if size_limit < len(filenames):
    filenames = filenames[:size_limit]

  if mode == 'classification':
    dataset_params_list[params_idx].label_per_step = False
  elif mode == 'semantic_segmentation':
    dataset_params_list[params_idx].label_per_step = True
  elif mode == 'self:triplets':
    dataset_params_list[params_idx].label_per_step = True

  for file in filenames:
    # f = tf.constant(file)
    yield load_data_classification(file, params_idx)

def check_tf_dataset():
  params = EasyDict()
  params.batch_size = 1
  params.seq_len = 200
  params.adjust_vertical_model = 1
  params.walk_alg = 'no_local_jumps'  # no_repeat / fastest / only_jumps / no_local_jumps
  params.net_input = ['xyz'] # edge_meshcnn , xyz , vertex_indices
  params.net_input += ['jump_indication', 'vertex_indices']
  params.n_walks_per_model = 2
  params.normalize_model = True
  params.sub_mean_for_data_augmentation = False
  params.reverse_walk = False
  params.train_data_augmentation = {'horisontal_90deg': [0, 1]}
  params.classes_indices_to_use = None
  params.uniform_starting_point = False
  params.network_task = 'semantic_segmentation'
  cpos = None
  v_size = None
  min_max_faces2use = [0, np.inf]
  if 1:
    pathname_expansion = '/home/ran/Databases/modelnet40_1k2k4k/*test*.npz'
    mode = 'classification'
  elif 0:
    pathname_expansion = os.path.expanduser('~') + '/runs/datasets_processed/human_seg_from_mcnn_with_nrmls/*train*.npz'
    mode = 'semantic_segmentation' # classification / semantic_segmentation
    cpos = [(-2.5, -0.12, 4.0), (0., 0., 0.1), (0., 1., 0.)]
  elif 0:
    pathname_expansion = os.path.expanduser('~') + '/runs/datasets_processed/shrec11_raw_4k/*.npz'
    mode = 'classification' # classification / semantic_segmentation
    params.seq_len = 500
    params.classes_indices_to_use = [0, 1, 2, 3, 4]  # [i for i in range(20)] # None

  if 0:
    dataset_iterator, n_items = tf_mesh_dataset(params, pathname_expansion=pathname_expansion, mode=mode,
                                                permute_file_names=True, shuffle_size=0, min_max_faces2use=min_max_faces2use)
  else:
    # TODO: test this
    dataset_iterator = mesh_dataset_iterator(params, pathname_expansion=pathname_expansion, mode=mode,
                                             permute_file_names=True, min_max_faces2use=min_max_faces2use)
  tb = time.time()
  n = 0
  for _ in range(3):
    for name, features, labels in dataset_iterator:
      print('   ', name, features.shape, labels)
      if 1:
      #for __ in range(1):
        if 0: # Show features
          for f in range(features.shape[-1]):
            plt.subplot(2, 2, f + 1)
            plt.plot(features[:,:,f].T)
        if 1:
          fn = name[name.find(':') + 1:]
          mesh_data = np.load(fn, encoding='latin1', allow_pickle=True)
          vertices = mesh_data['vertices']
          if 0:
            data_augmentation_axes_rot(vertices)
          if 0:
            model = {'vertices': vertices, 'faces': mesh_data['faces']}
            dataset_prepare.calc_mesh_area(model)
            v_size = model['area_vertices'] / model['area_vertices'].max() * 30
          v_colors = -1 * np.ones((vertices.shape[0])).astype(np.int)
          if 0: # Show labels
            v_colors = mesh_data['labels'].astype(np.int)
          elif 1: # Show walk
            v_colors = -1 * np.ones((vertices.shape[0])).astype(np.int)
            for wi in range(features.shape[0]):
              walk = features[wi, :, -1].astype(np.int)
              jumps = features[wi, :, -2].astype(np.int)
              v_colors[walk] = wi # mesh_data['labels']
              print(wi, ') % unique vertices: ', round(np.unique(walk).size / walk.size * 100, 2), 'n jumps: ', jumps.sum())
              utils.visualize_model_walk(vertices, mesh_data['faces'], walk, jumps, cpos=cpos)
          if 0:
            for i in range(10):
              v_colors = -1 * np.ones((vertices.shape[0])).astype(np.int)
              v_colors[mesh_data['labels'] == i] = 1
          utils.visualize_model(vertices, mesh_data['faces'], vertex_colors_idx=v_colors, cpos=cpos, v_size=v_size)
      n += 1
      #print(name)
  print(n, 'Time: ', time.time() - tb)

def check_triplet_walk():
  params = EasyDict()
  params.batch_size = 1
  params.seq_len = 20
  params.adjust_vertical_model = 0
  params.walk_alg = 'no_jumps'  # no_repeat / fastest / only_jumps / no_local_jumps / no_jumps
  params.net_input = ['xyz'] # edge_meshcnn , xyz , vertex_indices
  params.net_input += ['jump_indication', 'vertex_indices']
  params.n_walks_per_model = 4
  params.normalize_model = False
  params.sub_mean_for_data_augmentation = False
  params.reverse_walk = True
  params.train_data_augmentation = {}
  params.classes_indices_to_use = None
  params.uniform_starting_point = False
  params.network_task = 'self:triplets'
  cpos=[(-2.4, 2.8, 4.0), (0., 0.5, 0.0), (0., 1., 0.)]
  v_size = None
  min_max_faces2use = [0, np.inf]
  if 1:
    pathname_expansion = '/home/alon/runs/datasets_processed/human_seg_with_geodist/*.npz'
    mode = 'self:triplets'
  if 0:
    dataset_iterator, n_items = tf_mesh_dataset(params, pathname_expansion=pathname_expansion, mode=mode,
                                                permute_file_names=True, shuffle_size=0, min_max_faces2use=min_max_faces2use)
  else:
    dataset_iterator = mesh_dataset_iterator(params, pathname_expansion=pathname_expansion, mode=mode,
                                             permute_file_names=True, min_max_faces2use=min_max_faces2use)
  tb = time.time()
  for name, features, labels in dataset_iterator:
    if type(name) == type(tf.constant(0)):
      name = name.numpy()[0].decode()
      features = features.numpy()[0]
    fn = name[name.find(':') + 1:]
    mesh_data = np.load(fn, encoding='latin1', allow_pickle=True)
    vertices = mesh_data['vertices']
    if 1: # Show walks
      v_colors = -1 * np.ones((vertices.shape[0])).astype(np.int)
      for wi in range(features.shape[0]):
        walk = features[wi, :, -1].astype(np.int)
        jumps = features[wi, :, -2].astype(np.int)
        v_colors[walk] = wi # mesh_data['labels']
    utils.visualize_model(vertices, mesh_data['faces'], vertex_colors_idx=v_colors, cpos=cpos, v_size=v_size)
  print('Time: ', time.time() - tb)


if __name__ == '__main__':
  # utils.config_gpu(False)
  os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
  np.random.seed(1)
  check_tf_dataset()
  #check_triplet_walk()
  #check_fns()