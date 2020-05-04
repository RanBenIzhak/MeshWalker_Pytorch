import numpy as np

def norm_vertices(vertices, params):
  # Move the model so the bbox center will be at (0, 0, 0)
  mean = np.mean((np.min(vertices, axis=0), np.max(vertices, axis=0)), axis=0)
  vertices -= mean

  # Scale model to fit into the unit ball
  abs_max = np.max(vertices)
  vertices /= abs_max

  if params.sub_mean_for_data_augmentation:
    vertices -= np.nanmean(vertices, axis=0)


def fill_dxdydz_features(features, f_idx, vertices, mesh_extra, seq, jumps, seq_len):
  walk = np.diff(vertices[seq[:seq_len + 1]], axis=0) * 100
  features[:, f_idx:f_idx + walk.shape[1]] = walk
  f_idx += 3
  return f_idx


def fill_xyz_features(features, f_idx, vertices, mesh_extra, seq, jumps, seq_len):
  walk = vertices[seq[1:seq_len + 1]]
  features[:, f_idx:f_idx + walk.shape[1]] = walk
  f_idx += 3
  return f_idx


def load_npz(npz_fn, params):
  return_data = {}
  with np.load(npz_fn, encoding='latin1', allow_pickle=True) as mesh_data:
    return_data['vertices'] = mesh_data['vertices']
    return_data['faces'] = mesh_data['faces']
    return_data['edges'] = mesh_data['edges']
    if params.label_per_step:
      return_data['labels'] = mesh_data['labels']
    else:
      return_data['labels'] = mesh_data['label']
    if params.kdtree_query_needed:
      return_data['kdtree_query'] = mesh_data['kdtree_query']
    else:
      return_data['kdtree_query'] = [-1]

    return_data['name'] = mesh_data['dataset_name'].tolist() + ':' + npz_fn

  return return_data
