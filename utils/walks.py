# import tensorflow as tf

import numpy as np
from utils.mesh import norm_vertices

def generate_walk(mesh_data, dataset_params):
  vertices = mesh_data['vertices']
  seq_len = dataset_params.seq_len
  if dataset_params.normalize_model:
    norm_vertices(vertices, dataset_params)
  # # Get essential data from file
  # if dataset_params.label_per_step:
  #   mesh_labels = mesh_data['labels']
  # else:
  mesh_labels = -1 * np.ones((vertices.shape[0],))
  mesh_extra = {}
  mesh_extra['n_vertices'] = vertices.shape[0]
  if dataset_params.edges_needed:
    mesh_extra['edges'] = mesh_data['edges']
  if dataset_params.kdtree_query_needed:
    mesh_extra['kdtree_query'] = mesh_data['kdtree_query']
  features = np.zeros((dataset_params.n_walks_per_model, seq_len, dataset_params.number_of_features), dtype=np.float32)
  labels = np.zeros((dataset_params.n_walks_per_model, seq_len), dtype=np.int32)

  for walk_id in range(dataset_params.n_walks_per_model):
    f0 = np.random.randint(vertices.shape[0])
    seq, jumps = dataset_params.walk_function(mesh_extra, f0, seq_len)
    # seq, jumps = get_seq_random_walk_no_repeat(mesh_extra, f0, seq_len)

    f_idx = 0
    for fill_ftr_fun in dataset_params.fill_features_functions:
      f_idx = fill_ftr_fun(features[walk_id], f_idx, vertices, mesh_extra, seq, jumps, seq_len)
  return features, labels

def jump_to_closest_unviseted(model_kdtree_query, model_n_vertices, walk, enable_super_jump=True):
  for nbr in model_kdtree_query[walk[-1]]:
    if nbr not in walk:
      return nbr

  if not enable_super_jump:
    return None

  # If not fouind, jump to random node
  node = np.random.randint(model_n_vertices)

  return node

def get_seq_random_walk_no_repeat(mesh_extra, f0, seq_len, verbose=False):
  nbrs = mesh_extra['edges']
  n_vertices = mesh_extra['n_vertices']
  seq = np.zeros((seq_len + 1,), dtype=np.int32)
  jumps = np.zeros((seq_len + 1,), dtype=np.bool)
  if verbose:
    print(' ---- >>> f0: ', f0)
  seq[0] = f0
  jumps[0] = [True]
  backward_steps = 1
  for i in range(1, seq_len + 1):
    this_nbrs = nbrs[seq[i - 1]]
    if 0:
      nodes_to_consider = [n for n in this_nbrs if n != -1 and n not in seq[max(i - 20, 0):i]]
    elif 1:
      nodes_to_consider = [n for n in this_nbrs if n != -1 and n not in seq[:i]]
    if len(nodes_to_consider):
      to_add = np.random.choice(nodes_to_consider)
      jump = False
    else:
      to_add = jump_to_closest_unviseted(mesh_extra['kdtree_query'], n_vertices, seq[:i], enable_super_jump=False)
      if to_add is None:
        if i > backward_steps:
          to_add = seq[i - backward_steps - 1]
          backward_steps += 2
        else:
          to_add = np.random.randint(n_vertices)
          jump = True
      else:
        jump = True
    seq[i] = to_add
    jumps[i] = jump

  return seq, jumps

def get_seq_random_walk_no_jumps(mesh_extra, f0, seq_len, verbose=False, back_steps_no_rpt=np.inf):
  #tb = time.perf_counter()
  nbrs = mesh_extra['edges']
  n_vertices = mesh_extra['n_vertices']
  seq = np.zeros((seq_len + 1,), dtype=np.int32)
  jumps = np.zeros((seq_len + 1,), dtype=np.bool)
  visited = np.zeros((n_vertices + 1,), dtype=np.bool)
  visited[-1] = True
  visited[f0] = True
  seq[0] = f0
  jumps[0] = [True]
  backward_steps = 1
  for i in range(1, seq_len + 1):
    this_nbrs = nbrs[seq[i - 1]]
    if back_steps_no_rpt:
      b = max(0, i - back_steps_no_rpt)
    else:
      b = 0
    nodes_to_consider = [n for n in this_nbrs if not visited[n]]
    if 0:
      nodes_to_consider_ = [n for n in this_nbrs if n != -1 and n not in seq[b:i]]
      assert np.all(np.array(nodes_to_consider) == np.array(nodes_to_consider_))
    if len(nodes_to_consider):
      to_add = np.random.choice(nodes_to_consider)
      jump = False
    else:
      #if jumps[i - backward_steps - 1] == 0:
      if i > backward_steps:
        to_add = seq[i - backward_steps - 1]
        backward_steps += 2
      else:
        #raise Exception('jump is not allowed!')
        to_add = np.random.randint(n_vertices)
        jump = True
    seq[i] = to_add
    jumps[i] = jump
    visited[to_add] = 1

  #tf.print(time.perf_counter() - tb)

  return seq, jumps

def get_seq_random_walk_no_local_jumps(mesh_extra, f0, seq_len):
  nbrs = mesh_extra['edges']
  n_vertices = mesh_extra['n_vertices']
  seq = np.zeros((seq_len + 1, ), dtype=np.int32)
  jumps = np.zeros((seq_len + 1,), dtype=np.bool)
  seq[0] = f0
  jumps[0] = True
  for i in range(1, seq_len + 1):
    faces_to_consider = nbrs[seq[i - 1]]
    to_add = -1
    for m in range(16):
      to_add = np.random.choice(faces_to_consider)
      jump = False
      if to_add == -1:
        continue
      if to_add not in seq[max(0, i-20):i]:
        break
    if to_add == -1:
      to_add = np.random.randint(n_vertices)
      jump = True

    seq[i] = to_add
    jumps[i] = jump

  return seq, jumps

def get_seq_random_walk_fast(mesh_extra, f0, seq_len):
  # --> fixed - to check at ModelNet40!
  nbrs = mesh_extra['edges']
  kdtr = mesh_extra['kdtree_query']
  n_vertices = mesh_extra['n_vertices']
  seq = np.zeros((seq_len + 1, ), dtype=np.int32)
  jumps = np.zeros((seq_len + 1,), dtype=np.bool)
  seq[0] = f0
  jumps[0] = True
  for i in range(1, seq_len + 1):
    faces_to_consider = nbrs[seq[i - 1]]
    to_add = -1
    for m in range(6):
      to_add = np.random.choice(faces_to_consider)
      jump = False
      if to_add == -1:# and m > 4:
        to_add = np.random.choice(kdtr[seq[i - 1]])
        jump = True
      if to_add not in seq[max(0, i-20):i]:
        break
    if to_add == -1:
      to_add = np.random.randint(n_vertices)
      jump = True

    seq[i] = to_add
    jumps[i] = jump

  return seq, jumps

def get_seq_random_walk_fastest(mesh_extra, f0, seq_len):
  nbrs = mesh_extra['edges']
  n_vertices = mesh_extra['n_vertices']
  seq = np.zeros((seq_len + 1, ), dtype=np.int32)
  jumps = np.zeros((seq_len + 1,), dtype=np.bool)
  seq[0] = f0
  jumps[0] = True
  for i in range(1, seq_len + 1):
    faces_to_consider = nbrs[seq[i - 1]]
    to_add = np.random.choice(faces_to_consider)
    jump = False
    if to_add == -1:
      to_add = np.random.randint(n_vertices)
      jump = True
    seq[i] = to_add
    jumps[i] = jump

  return seq, jumps

def get_seq_random_walk_local_jumps(mesh_extra, f0, seq_len):
  n_vertices = mesh_extra['n_vertices']
  kdtr = mesh_extra['kdtree_query']
  seq = np.zeros((seq_len + 1, ), dtype=np.int32)
  jumps = np.zeros((seq_len + 1,), dtype=np.bool)
  seq[0] = f0
  for i in range(1, seq_len + 1):
    b = min(0, i - 20)
    to_consider = [n for n in kdtr[seq[i - 1]] if n != -1 and n not in seq[b:i]]
    if len(to_consider):
      seq[i] = np.random.choice(to_consider)
      jumps[i] = False
    else:
      seq[i] = np.random.randint(n_vertices)
      jumps[i] = True

  return seq, jumps

def get_seq_random_walk_fastest_only_jumps(mesh_extra, f0, seq_len):
  n_vertices = mesh_extra['n_vertices']
  seq = np.zeros((seq_len + 1, ), dtype=np.int32)
  jumps = np.ones((seq_len + 1,), dtype=np.bool)
  seq[0] = f0
  for i in range(1, seq_len + 1):
    seq[i] = np.random.randint(n_vertices)

  return seq, jumps


def debug_walk():
  ''' Testing get random walk for mesh'''
  # step A - load example shapenet mesh
  npz_path = '/home/ran/PycharmProjects/MeshWalker_Pytorch/Data/shapenet_tmp/1a6f615e8b1b5ae4dbbc9440457e303e_simplified_to_1000.npz'
  from datasets.dataset_directory import load_model_from_npz as load_npz
  mesh_data = load_npz(npz_path)
  from utils.parameters import shapenet_params
  params = shapenet_params()
  print('debug break 1')

  test = generate_walk(mesh_data, params)
  print('working')