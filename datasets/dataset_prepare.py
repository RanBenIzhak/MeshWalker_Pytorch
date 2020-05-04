import time, glob, os, shutil, sys, json
from pathlib import Path

import trimesh
import open3d   # this cannot be imported AFTER pytorch, need to import in main script
import pyvista as pv
from easydict import EasyDict
import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm
# import gdist

from utils import utils, walks, mesh_prepare__from_meshcnn
from datasets import dataset_directory

sigg17_part_labels = ['---', 'head', 'hand', 'lower-arm', 'upper-arm', 'body', 'upper-lag', 'lower-leg', 'foot']
sigg17_shape2label = {v: k for k, v in enumerate(sigg17_part_labels)}

dancer_part_labels = ['head',           'chest',          'abdomen',          'pelvis',           'left upper arm',
                      'left lower arm', 'left hand',      'right upper arm',  'right lower arm',  'right hand',
                      'left upper leg', 'left lower leg', 'left foot',        'right upper leg',  'right lower leg',
                      'right foot']

model_net_labels = [
  'bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet',
  'wardrobe', 'bookshelf', 'laptop', 'door', 'lamp', 'person', 'curtain', 'piano', 'airplane', 'cup',
  'cone', 'tent', 'radio', 'stool', 'range_hood', 'car', 'sink', 'guitar', 'tv_stand', 'stairs',
  'mantel', 'bench', 'plant', 'bottle', 'bowl', 'flower_pot', 'keyboard', 'vase', 'xbox', 'glass_box'
]
model_net_shape2label = {v: k for k, v in enumerate(model_net_labels)}

cubes_labels = [
  'apple',  'bat',      'bell',     'brick',      'camel',
  'car',    'carriage', 'chopper',  'elephant',   'fork',
  'guitar', 'hammer',   'heart',    'horseshoe',  'key',
  'lmfish', 'octopus',  'shoe',     'spoon',      'tree',
  'turtle', 'watch'
]
cubes_shape2label = {v: k for k, v in enumerate(cubes_labels)}

shrec11_labels = [
  'armadillo',  'man',      'centaur',    'dinosaur',   'dog2',
  'ants',       'rabbit',   'dog1',       'snake',      'bird2',
  'shark',      'dino_ske', 'laptop',     'santa',      'flamingo',
  'horse',      'hand',     'lamp',       'two_balls',  'gorilla',
  'alien',      'octopus',  'cat',        'woman',      'spiders',
  'camel',      'pliers',   'myScissor',  'glasses',    'bird1'
]
shrec11_shape2label = {v: k for k, v in enumerate(shrec11_labels)}

coseg_labels = [
  '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c',
]
coseg_shape2label = {v: k for k, v in enumerate(coseg_labels)}

def calc_mesh_area(mesh):
  t_mesh = trimesh.Trimesh(vertices=mesh['vertices'], faces=mesh['faces'], process=False)
  mesh['area_faces'] = t_mesh.area_faces
  mesh['area_vertices'] = np.zeros((mesh['vertices'].shape[0]))
  for f_index, f in enumerate(mesh['faces']):
    for v in f:
      mesh['area_vertices'][v] += mesh['area_faces'][f_index] / f.size

def calc_vertex_labels_from_face_labels(mesh, face_labels):
  vertices = mesh['vertices']
  faces = mesh['faces']
  all_vetrex_labels = [[] for _ in range(vertices.shape[0])]
  vertex_labels = -np.ones((vertices.shape[0],), dtype=np.int)
  for i in range(faces.shape[0]):
    label = face_labels[i]
    for f in faces[i]:
      all_vetrex_labels[f].append(label)
  for i in range(vertices.shape[0]):
    counts = np.bincount(all_vetrex_labels[i])
    vertex_labels[i] = np.argmax(counts)
  return vertex_labels


def prepare_cache_walk(mesh, seq_len):
  vertices = mesh['vertices']
  assert vertices.shape[0] < 2 ** 16
  _n_walks_cahce = 4
  model = EasyDict({'vertices': mesh['vertices'], 'faces': mesh['faces'], 'edges': mesh['edges'],
                    'kdtree_query': mesh['kdtree_query']})
  model.node_pos = model.vertices
  model.walk_cache = []
  model.walk_cache_jumps = []
  for _ in range(vertices.shape[0]):
    model.walk_cache.append(65535 * np.ones((_n_walks_cahce, seq_len + 2), dtype=np.uint16))
    model.walk_cache_jumps.append(np.zeros((_n_walks_cahce, seq_len + 2), dtype=np.bool))
  model.use_cache_walk = True
  for f_start in range(vertices.shape[0]):
    for cach_idx in range(_n_walks_cahce):
      walks.get_seq_random_walk_no_repeat(model, f_start, seq_len, cach_idx=cach_idx)
  mesh['walk_cache'] = model.walk_cache
  mesh['walk_cache_jumps'] = model.walk_cache_jumps


def prepare_meshcnn_features(mesh_data, mesh_extra):
  mesh = EasyDict({'vertices': mesh_data['vertices'], 'faces': mesh_data['faces']})
  edges_np, edge_features = mesh_prepare__from_meshcnn.calc_edge_features(mesh)
  zero_ftrs = np.zeros((1, edge_features.T.shape[1]))  # The 1st feature is 0, to be used for step which is not an edge
  mesh_extra['edge_features'] = np.vstack((zero_ftrs, edge_features.T))

  row_ind = []; col_ind = []; data = []
  for i, e in enumerate(edges_np):
    row_ind.append(e[0])
    col_ind.append(e[1])
    data.append(i + 1)
    row_ind.append(e[1])
    col_ind.append(e[0])
    data.append(i + 1)
  nv = mesh['vertices'].shape[0]
  mesh_extra['edges_map'] = csr_matrix((data, (row_ind, col_ind)), shape=(nv, nv))

def prepare_edges_and_kdtree(mesh):
  vertices = mesh['vertices']
  faces = mesh['faces']
  mesh['edges'] = [set() for _ in range(vertices.shape[0])]
  for i in range(faces.shape[0]):
    for v in faces[i]:
      mesh['edges'][v] |= set(faces[i])
  for i in range(vertices.shape[0]):
    if i in mesh['edges'][i]:
      mesh['edges'][i].remove(i)
    mesh['edges'][i] = list(mesh['edges'][i])
  max_vertex_degree = np.max([len(e) for e in mesh['edges']])
  for i in range(vertices.shape[0]):
    if len(mesh['edges'][i]) < max_vertex_degree:
      mesh['edges'][i] += [-1] * (max_vertex_degree - len(mesh['edges'][i]))
  mesh['edges'] = np.array(mesh['edges'], dtype=np.int32)
  mesh['kdtree_query'] = []
  t_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
  n_nbrs = min(10, vertices.shape[0] - 2)
  for n in range(vertices.shape[0]):
    d, i_nbrs = t_mesh.kdtree.query(vertices[n], n_nbrs)
    i_nbrs_cleared = [inbr for inbr in i_nbrs if inbr != n and inbr < vertices.shape[0]]
    if len(i_nbrs_cleared) > n_nbrs - 1:
      i_nbrs_cleared = i_nbrs_cleared[:n_nbrs - 1]
    mesh['kdtree_query'].append(np.array(i_nbrs_cleared, dtype=np.int32))
  mesh['kdtree_query'] = np.array(mesh['kdtree_query'])
  assert mesh['kdtree_query'].shape[1] == (n_nbrs - 1), 'Number of kdtree_query is wrong: ' + str(mesh['kdtree_query'].shape[1])

  mesh['vertex_normals'] = t_mesh.vertex_normals

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

def prepare_geo_dist(mesh, max_distance=None):
  MAX_CANDIDATES = 20
  if max_distance is None:
    mx1, mx2, mx3 = np.max(mesh['vertices'], axis=0)
    mn1, mn2, mn3 = np.min(mesh['vertices'], axis=0)
    max_distance = max(mx1 - mn1, mx2 - mn2, mx3 - mn3) * 0.5
    mid_dist = max_distance / 2
  geo_dist = gdist.local_gdist_matrix(mesh['vertices'], mesh['faces'], max_distance=max_distance)
  mesh['geo_dist'] = geo_dist
  mid_vertices = []
  far_vertices = []
  for v in range(mesh['vertices'].shape[0]):
    not_far_v = geo_dist[v].nonzero()[1]
    a = np.array(geo_dist[v, not_far_v].todense()).flatten()
    mid_range_idxs = not_far_v[np.where(a > mid_dist)[0]]
    mid_ver = np.random.permutation(mid_range_idxs)[:MAX_CANDIDATES]
    mid_vertices.append(mid_ver)

    far_v = np.where(geo_dist[v].todense() == 0)[1]
    far_v = np.random.permutation(far_v)[:MAX_CANDIDATES]
    far_vertices.append(far_v)
  mesh['mid_vertices'] = np.array(mid_vertices)
  mesh['far_vertices'] = np.array(far_vertices)

def add_fields_and_dump_model(mesh_data, fileds_needed, out_fn, dataset_name, dump_model=True):
  m = {}
  for k, v in mesh_data.items():
    if k in fileds_needed:
      m[k] = v
  for field in fileds_needed:
    if field not in m.keys():
      if field == 'edge_features' or field == 'edges_map':
        prepare_meshcnn_features(m, m)
      if field == 'labels':
        m[field] = np.zeros((0,))
      if field == 'dataset_name':
        m[field] = dataset_name
      if field == 'walk_cache':
        m[field] = np.zeros((0,))
      if field == 'kdtree_query' or field == 'edges':
        try:
          prepare_edges_and_kdtree(m)
        except:
          print('debug')
      if field == 'walk_cache':
        prepare_cache_walk(m, 200)
      if field == 'geo_dist':
        prepare_geo_dist(m)
      if field == 'area_vertices_list':
        v_tmp = m['vertices'].copy()
        mean = np.mean((np.min(v_tmp, axis=0), np.max(v_tmp, axis=0)), axis=0)
        v_tmp -= mean
        abs_max = np.max(v_tmp)
        v_tmp /= abs_max
        m['area_vertices_list'] = prepare_area_vertices_list(v_tmp)

  if dump_model:
    np.savez(out_fn, **m)

  return m

def get_mesh_from_point_cloud(path, n_target_faces):
  transform_names = {'cutting_instrument': 'knife'}
  points10k = np.loadtxt(path + '/point_sample/pts-10000.txt')
  labels10k = np.loadtxt(path + '/point_sample/label-10000.txt')

  if 0:
    from sklearn.cluster import KMeans
    p = pv.Plotter()
    p.add_mesh(pv.PolyData(points10k), point_size=5, render_points_as_spheres=True)
    kmeans = KMeans(n_clusters=10).fit(points10k)
    p.add_mesh(pv.PolyData(kmeans.cluster_centers_), point_size=50, color='r', render_points_as_spheres=True)
    p.show()

  with open(path + '/result_after_merging.json') as json_file:
    h_tree = json.load(json_file)

  with open(path + '/meta.json') as json_file:
    meta = json.load(json_file)

  model_label = h_tree[0]['id']
  model_name  = h_tree[0]['name']
  if model_name in transform_names.keys():
    model_name = transform_names[model_name]

  radius = 0.05
  pcd = open3d.geometry.PointCloud()
  pcd.points = open3d.utility.Vector3dVector(points10k)
  radius_normal = radius
  kd_search_params = open3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
  pcd.estimate_normals(search_param=kd_search_params)
  mesh_full = open3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting\
    (pcd, open3d.utility.DoubleVector([radius, radius * 2]))

  for t_faces in np.sort(n_target_faces)[::-1]:
    mesh = mesh_full.simplify_quadric_decimation(t_faces)
    mesh = mesh.remove_unreferenced_vertices()
    labels = fix_labels_by_dist(np.asarray(mesh.vertices), points10k, labels10k)
    yield mesh, labels, t_faces, model_name, model_label

  if 0:
    open3d.visualization.draw_geometries([pcd])
    open3d.visualization.draw_geometries([mesh])

def get_partnet_labels(path):
  transform_names = {'cutting_instrument': 'knife'}
  points10k = np.loadtxt(path + '/point_sample/pts-10000.txt')
  labels10k = np.loadtxt(path + '/point_sample/label-10000.txt')

  with open(path + '/result_after_merging.json') as json_file:
    h_tree = json.load(json_file)

  with open(path + '/meta.json') as json_file:
    meta = json.load(json_file)

  radius = 0.1
  pcd = open3d.geometry.PointCloud()
  pcd.points = open3d.utility.Vector3dVector(points10k)
  voxel_size = 0.05
  radius_normal = voxel_size * 2  # from NADAV: from triangles -> w.avarage of them
  kd_search_params = open3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
  pcd.estimate_normals(search_param=kd_search_params)
  mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting\
    (pcd, open3d.utility.DoubleVector([radius, radius * 2]))
  open3d.visualization.draw_geometries([pcd])
  open3d.visualization.draw_geometries([mesh])

  model_label = h_tree[0]['id']
  model_name  = h_tree[0]['name']
  if model_name in transform_names.keys():
    model_name = transform_names[model_name]

  return model_name, model_label, points10k, labels10k

def get_sig17_seg_bm_labels(mesh, file, seg_path):
  # Finding the best match file name .. :
  in_to_check = file.replace('obj', 'txt')
  in_to_check = in_to_check.replace('off', 'txt')
  in_to_check = in_to_check.replace('_fix_orientation', '')
  if in_to_check.find('MIT_animation') != -1 and in_to_check.split('/')[-1].startswith('mesh_'):
    in_to_check = '/'.join(in_to_check.split('/')[:-2])
    in_to_check = in_to_check.replace('MIT_animation/meshes_', 'mit/mit_')
    in_to_check += '.txt'
  elif in_to_check.find('/scape/') != -1:
    in_to_check = '/'.join(in_to_check.split('/')[:-1])
    in_to_check += '/scape.txt'
  elif in_to_check.find('/faust/') != -1:
    in_to_check = '/'.join(in_to_check.split('/')[:-1])
    in_to_check += '/faust.txt'

  seg_full_fn = []
  for fn in Path(seg_path).rglob('*.txt'):
    tmp = str(fn)
    tmp = tmp.replace('/segs/', '/meshes/')
    tmp = tmp.replace('_full', '')
    tmp = tmp.replace('shrec_', '')
    tmp = tmp.replace('_corrected', '')
    if tmp == in_to_check:
      seg_full_fn.append(str(fn))
  if len(seg_full_fn) == 1:
    seg_full_fn = seg_full_fn[0]
  else:
    print('\nin_to_check', in_to_check)
    print('tmp', tmp)
    raise Exception('!!')
  face_labels = np.loadtxt(seg_full_fn)

  return face_labels


def get_labels(dataset_name, mesh, file, fn2labels_map=None):
  if dataset_name == 'faust':
    face_labels = np.load('faust_labels/faust_part_segmentation.npy').astype(np.int)
    vertex_labels = calc_vertex_labels_from_face_labels(mesh, face_labels)
    model_label = np.zeros((0,))
    return model_label, vertex_labels
  elif dataset_name.startswith('coseg') or dataset_name == 'human_seg_from_meshcnn':
    labels_fn = '/'.join(file.split('/')[:-2]) + '/seg/' + file.split('/')[-1].split('.')[-2] + '.eseg'
    e_labels = np.loadtxt(labels_fn)
    v_labels = [[] for _ in range(mesh['vertices'].shape[0])]
    faces = mesh['faces']

    edge2key = dict()
    edges = []
    edges_count = 0
    for face_id, face in enumerate(faces):
      faces_edges = []
      for i in range(3):
        cur_edge = (face[i], face[(i + 1) % 3])
        faces_edges.append(cur_edge)
      for idx, edge in enumerate(faces_edges):
        edge = tuple(sorted(list(edge)))
        faces_edges[idx] = edge
        if edge not in edge2key:
          edge2key[edge] = edges_count
          edges.append(list(edge))
          v_labels[edge[0]].append(e_labels[edges_count])
          v_labels[edge[1]].append(e_labels[edges_count])
          edges_count += 1

    vertex_labels = []
    for l in v_labels:
      l2add = np.argmax(np.bincount(l))
      vertex_labels.append(l2add)
    vertex_labels = np.array(vertex_labels)
    model_label = np.zeros((0,))
    return model_label, vertex_labels
  else:
    tmp = file.split('/')[-1]
    model_name = '_'.join(tmp.split('_')[:-1])
    if dataset_name.lower().startswith('modelnet'):
      model_label = model_net_shape2label[model_name]
    elif dataset_name.lower().startswith('cubes'):
      model_label = cubes_shape2label[model_name]
    elif dataset_name.lower().startswith('shrec11'):
      model_name = file.split('/')[-3]
      if fn2labels_map is None:
        model_label = shrec11_shape2label[model_name]
      else:
        file_index = int(file.split('.')[-2].split('T')[-1])
        model_label = fn2labels_map[file_index]
    else:
      raise Exception('Cannot find labels for the dataset')
    vertex_labels = np.zeros((0,))
    return model_label, vertex_labels

def fix_labels_by_dist(vertices, orig_vertices, labels_orig):
  labels = -np.ones((vertices.shape[0], ))

  for i, vertex in enumerate(vertices):
    d = np.linalg.norm(vertex - orig_vertices, axis=1)
    orig_idx = np.argmin(d)
    labels[i] = labels_orig[orig_idx]

  return labels

def remesh(mesh_orig, target_n_faces, add_labels=False, labels_orig=None):
  labels = labels_orig
  if target_n_faces < np.asarray(mesh_orig.triangles).shape[0]:
    mesh = mesh_orig.simplify_quadric_decimation(target_n_faces)
    str_to_add = '_simplified_to_' + str(target_n_faces)
    mesh = mesh.remove_unreferenced_vertices()
    if add_labels and labels_orig.size:
      labels = fix_labels_by_dist(np.asarray(mesh.vertices), np.asarray(mesh_orig.vertices), labels_orig)
  else:
    mesh = mesh_orig
    mesh = mesh.remove_unreferenced_vertices()
    if add_labels and labels_orig.size:
      labels = fix_labels_by_dist(np.asarray(mesh.vertices), np.asarray(mesh_orig.vertices), labels_orig)
    str_to_add = '_not_changed_' + str(np.asarray(mesh_orig.triangles).shape[0])

  return mesh, labels, str_to_add

def load_meshes(model_fns):
  f_names = glob.glob(model_fns)
  joint_mesh_vertices = []
  joint_mesh_faces = []
  for fn in f_names:
    if not fn.endswith('.obj'):
      continue
    mesh_ = trimesh.load_mesh(fn, file_type='obj')
    mesh_ = utils.as_mesh(mesh_)
    vertex_offset = len(joint_mesh_vertices)
    joint_mesh_vertices += mesh_.vertices.tolist()
    faces = mesh_.faces + vertex_offset
    joint_mesh_faces += faces.tolist()

  mesh = open3d.geometry.TriangleMesh()
  mesh.vertices = open3d.utility.Vector3dVector(joint_mesh_vertices)
  mesh.triangles = open3d.utility.Vector3iVector(joint_mesh_faces)

  return mesh


def load_mesh(model_fn):
  if 1:  # To load and clean up mesh - "remove vertices that share position"
    mesh_ = trimesh.load_mesh(model_fn)
    mesh = open3d.geometry.TriangleMesh()
    mesh.vertices = open3d.utility.Vector3dVector(mesh_.vertices)
    mesh.triangles = open3d.utility.Vector3iVector(mesh_.faces)
  else:
    mesh = open3d.io.read_triangle_mesh(model_fn)

  return mesh

def get_model_and_ftrs(model_fn, params, target_n_faces=1000, dataset='modelnet'):
  fileds_needed = ['vertices', 'faces', 'edge_features', 'edges_map', 'edges', 'kdtree_query',
                   'label', 'labels', 'dataset_name', 'area_vertices_list']

  mesh_orig = load_mesh(model_fn)
  mesh, _, _ = remesh(mesh_orig, target_n_faces)

  mesh_data = EasyDict({'vertices': np.asarray(mesh.vertices), 'faces': np.asarray(mesh.triangles), 'n_faces_orig': np.asarray(mesh_orig.triangles).shape[0]})
  if dataset is None:
    label = -1
  else:
    label, labels = get_labels(dataset, mesh_data, model_fn)
  mesh_data = add_fields_and_dump_model(mesh_data, fileds_needed, None, 'modelnet40', dump_model=False)

  features = dataset_directory.mesh_data_to_walk_features(mesh_data, params)

  return mesh_data, features, label

def prepare_directory_segmentation(dataset_name, pathname_expansion, p_out, add_labels, fn_prefix, n_target_faces):
  fileds_needed = ['vertices', 'faces', 'edge_features', 'edges_map', 'edges', 'kdtree_query',
                   'label', 'labels', 'dataset_name', 'area_vertices_list']
  if not os.path.isdir(p_out):
    os.makedirs(p_out)
  filenames = glob.glob(pathname_expansion)
  filenames.sort()
  for file in tqdm(filenames):
    out_fn = p_out + '/' + fn_prefix + os.path.split(file)[1].split('.')[0]
    mesh = mesh_orig = load_mesh(file)
    mesh_data = EasyDict({'vertices': np.asarray(mesh.vertices), 'faces': np.asarray(mesh.triangles)})
    label, labels_orig = get_labels(dataset_name, mesh_data, file)
    for this_target_n_faces in n_target_faces:
      mesh, labels, str_to_add = remesh(mesh_orig, this_target_n_faces, add_labels=add_labels, labels_orig=labels_orig)
      mesh_data = EasyDict({'vertices': np.asarray(mesh.vertices), 'faces': np.asarray(mesh.triangles), 'label': label, 'labels': labels})
      out_fc_full = out_fn + str_to_add
      if os.path.isfile(out_fc_full + '.npz'):
        continue
      add_fields_and_dump_model(mesh_data, fileds_needed, out_fc_full, dataset_name)
      if 0:
        utils.visualize_model(mesh_data['vertices'], mesh_data['faces'], vertex_colors_idx=mesh_data['labels'].astype(np.int),
                              cpos=[(-2., -0.2, 3.3), (0., -0.3, 0.1), (0., 1., 0.)])
      if 0:
        utils.visualize_model(mesh_data['vertices'], mesh_data['faces'])#, cpos=[(-2., -0.2, 3.3), (0., -0.3, 0.1), (0., 1., 0.)])

def fix_mesh_human_body_from_meshcnn(mesh, model_name, verbose=False):
  vertices = np.asarray(mesh.vertices)
  flip_orientation_fn = ['test/shrec__7.obj', 'test/shrec__8.obj', 'test/shrec__9.obj', 'test/shrec__1.obj',
                         'test/shrec__11.obj', 'test/shrec__12.obj']
  if np.any([model_name.endswith(to_check) for to_check in flip_orientation_fn]):
    if verbose:
      print('\n\nOrientation changed\n\n')
    vertices = vertices[:, [0, 2, 1]]
  if model_name.find('/scape/') != -1:
    if verbose:
      print('\n\nOrientation changed 2\n\n')
    vertices = vertices[:, [1, 0, 2]]
  if model_name.endswith('test/shrec__12.obj'):
    if verbose:
      print('\n\nScaling factor 10\n\n')
    vertices = vertices / 10
  if model_name.find('/adobe') != -1:
    if verbose:
      print('\n\nScaling factor 100\n\n')
    vertices = vertices / 100

  # Fix so model minimum hieght will be 0 (person will be on the floor). Up is dim 1 (2nd)
  vertices[:, 1] -= vertices[:, 1].min()
  mesh.vertices = open3d.utility.Vector3dVector(vertices)

  return mesh

def prepare_directory_from_scratch(dataset_name, pathname_expansion=None, p_out=None, n_target_faces=None, add_labels=True,
                                   size_limit=np.inf, fn_prefix='', extra_fields_needed=[], fix_mesh_fn=None):
  if n_target_faces is None:
    n_target_faces = [np.inf, 8000, 4000, 2000]

  fileds_needed = ['vertices', 'faces', 'edge_features', 'edges_map', 'edges', 'kdtree_query',
                   'label', 'labels', 'dataset_name', 'area_vertices_list', 'vertex_normals']
  #fileds_needed += ['walk_cache', 'walk_cache_jumps']
  fileds_needed += extra_fields_needed

  if not os.path.isdir(p_out):
    os.makedirs(p_out)

  filenames = glob.glob(pathname_expansion)
  filenames.sort()
  if len(filenames) > size_limit:
    filenames = filenames[:size_limit]
  for file in tqdm(filenames):
    out_fn = p_out + '/' + fn_prefix + os.path.split(file)[1].split('.')[0]
    mesh = load_mesh(file)
    if fix_mesh_fn is not None:
      mesh = fix_mesh_fn(mesh, file)
    mesh_orig = mesh
    mesh_data = EasyDict({'vertices': np.asarray(mesh.vertices), 'faces': np.asarray(mesh.triangles)})
    if add_labels:
      if type(add_labels) is list:
        fn2labels_map = add_labels
      else:
        fn2labels_map = None
      label, labels_orig = get_labels(dataset_name, mesh_data, file, fn2labels_map=fn2labels_map)
    else:
      label = np.zeros((0, ))
    for this_target_n_faces in n_target_faces:
      mesh, labels, str_to_add = remesh(mesh_orig, this_target_n_faces, add_labels=add_labels, labels_orig=labels_orig)
      mesh_data = EasyDict({'vertices': np.asarray(mesh.vertices), 'faces': np.asarray(mesh.triangles), 'label': label, 'labels': labels})
      out_fc_full = out_fn + str_to_add
      if os.path.isfile(out_fc_full + '.npz'):
        continue
      m = add_fields_and_dump_model(mesh_data, fileds_needed, out_fc_full, dataset_name)
      if 0:
        utils.visualize_model(mesh_data['vertices'], mesh_data['faces'], vertex_colors_idx=mesh_data['labels'].astype(np.int))#,
                              #cpos=[(-2., -0.2, 3.3), (0., -0.3, 0.1), (0., 1., 0.)])
      if 0:
        utils.visualize_model(mesh_data['vertices'], mesh_data['faces'], title=file, cpos=[(-2.4, 2.8, 4.0), (0., 0.5, 0.0), (0., 1., 0.)])
      if 0: # Visualize geo-dist
        f0 = 30
        clrs = -np.ones((mesh_data['vertices'].shape[0],))
        clrs[f0] = 1
        clrs[m['geo_dist'][f0].nonzero()[1]] = 2
        utils.visualize_model(mesh_data['vertices'], mesh_data['faces'], vertex_colors_idx=clrs.astype(np.int))
      if 0: # visualize mid-far indices
        f0 = 30
        clrs = -np.ones((mesh_data['vertices'].shape[0],))
        clrs[f0] = 1
        clrs[m['mid_vertices'][f0]] = 2
        clrs[m['far_vertices'][f0]] = 3
        utils.visualize_model(mesh_data['vertices'], mesh_data['faces'], vertex_colors_idx=clrs.astype(np.int), title=file, cpos=[(-2., -0.2, 3.3), (0., -0.3, 0.1), (0., 1., 0.)])

def get_dataset_name(file):
  if os.path.split(file)[1].startswith('ModelNet'):
    return 'ModelNet'
  if file.find('dancer') != -1:
    return 'dancer'
  return 'unknown'


def copy_database_folder():
  if 1:
    p_in = '/home/alonlahav/runs/cache/dancer_full/'
  if 0:
    p_in = '/home/alonlahav/runs/cache/walk_len_102__faces_num_1000-2/'
  p_out = '/home/alonlahav/runs/cache/tmp/'
  fileds_needed = ['vertices', 'faces', 'edge_features', 'edges_map', 'edges', 'kdtree_query',
                   'walk_cache', 'walk_cache_jumps', 'label', 'labels', 'dataset_name', 'area_vertices_list']
  if os.path.isdir(p_out):
    shutil.rmtree(p_out)
  os.makedirs(p_out)

  pathname_expansion = p_in + '*.npz'
  filenames = glob.glob(pathname_expansion)
  for file in tqdm(filenames):
    mesh_data = np.load(file, encoding='latin1', allow_pickle=True)
    m = {}
    for k, v in mesh_data.items():
      if k in fileds_needed:
        m[k] = v
    for field in fileds_needed:
      if field not in mesh_data.keys():
        if field == 'edge_features' or field == 'edges_map':
          prepare_meshcnn_features(m, m)
        if field == 'labels':
          m[field] = np.zeros((0,))
        if field == 'dataset_name':
          m[field] = get_dataset_name(file)
    np.savez(p_out + '/' + os.path.split(file)[1], **m)

def prepare_modelnet40(part='all', size_limit=np.inf, fast_mode=False, n_target_faces=[1000, 2000, 4000]):
  if fast_mode:
    labels2use = model_net_labels[:4]
    n_target_faces = [1000]
    size_limit = 50
  elif part == 'all':
    labels2use = model_net_labels
  else:
    labels2use = model_net_labels[part * 10:part * 10 + 10]

  for i, name in enumerate(labels2use):
    print('-->>>', name)
    for part in ['test', 'train']:
      pin = os.path.expanduser('~') + '/datasets/ModelNet40/' + name + '/' + part + '/'
      prepare_directory_from_scratch('modelnet40', pathname_expansion=pin + '*.off',
                                     p_out=os.path.expanduser('~') + '/runs/datasets_processed/modelnet40_tmp/',
                                     add_labels='modelnet', n_target_faces=n_target_faces,
                                     size_limit=size_limit, fn_prefix=part + '_')

def prepare_part_net_using_mesh():
  partnet_path = '/media/alonlahav/4T-a/datasets/part_net/'
  out_path = os.path.expanduser('~') + '/runs/datasets_processed/part_net_tmp/'
  this_target_n_faces = 1000
  dataset_name = 'partnet'
  fileds_needed = ['vertices', 'faces', 'edge_features', 'edges_map', 'edges', 'kdtree_query',
                   'label', 'labels', 'dataset_name', 'area_vertices_list']

  if not os.path.isdir(out_path):
    os.makedirs(out_path)

  for d in tqdm(os.listdir(partnet_path)):
    mesh_orig = load_meshes(partnet_path + '/' + d + '/objs/*.obj')
    model_name, label, points10k, labels10k = get_partnet_labels(partnet_path + '/' + d)
    labels_orig = fix_labels_by_dist(np.asarray(mesh_orig.vertices), points10k, labels10k)
    mesh, labels, str_to_add = remesh(mesh_orig, this_target_n_faces, add_labels=True, labels_orig=labels_orig)
    mesh_data = EasyDict(
      {'vertices': np.asarray(mesh.vertices), 'faces': np.asarray(mesh.triangles), 'label': label, 'labels': labels})
    out_fc_full = out_path + '/' + model_name + '_' + d + str_to_add
    if os.path.isfile(out_fc_full + '.npz'):
      continue
    add_fields_and_dump_model(mesh_data, fileds_needed, out_fc_full, dataset_name)
    if 0:
      utils.visualize_model(mesh_data['vertices'], mesh_data['faces'], vertex_colors_idx=mesh_data['labels'].astype(np.int),
                            cpos=[(-2., -0.2, 3.3), (0., -0.3, 0.1), (0., 1., 0.)])


def prepare_part_net_using_points():
  partnet_path = '/media/alonlahav/4T-a/datasets/part_net/'
  out_path = os.path.expanduser('~') + '/runs/datasets_processed/part_net_points2mesh_tmp/'
  target_n_faces = [1000, 2000, 4000]
  dataset_name = 'partnet'
  fileds_needed = ['vertices', 'faces', 'edge_features', 'edges_map', 'edges', 'kdtree_query',
                   'label', 'labels', 'dataset_name', 'area_vertices_list']

  if not os.path.isdir(out_path):
    os.makedirs(out_path)

  for d in tqdm(os.listdir(partnet_path)):
    mesh_iter = get_mesh_from_point_cloud(partnet_path + '/' + d, target_n_faces)
    for mesh, labels, t_faces, model_name, model_label in mesh_iter:
      mesh_data = EasyDict({'vertices': np.asarray(mesh.vertices),
                            'faces': np.asarray(mesh.triangles),
                            'label': model_label,
                            'labels': labels})
      out_fc_full = out_path + '/' + model_name + '_' + d + '_simplified_to_' + str(t_faces)
      if os.path.isfile(out_fc_full + '.npz'):
        continue
      add_fields_and_dump_model(mesh_data, fileds_needed, out_fc_full, dataset_name)
      if 0:
        utils.visualize_model(mesh_data['vertices'], mesh_data['faces'], vertex_colors_idx=mesh_data['labels'].astype(np.int),
                              cpos=[(-2., -0.2, 3.3), (0., -0.3, 0.1), (0., 1., 0.)])


def prepare_shapenet_chairs_using_mesh():
  sn_chairs_path = '/home/ran/Databases/ShapeNetCore.v1/03001627'
  out_path = '../Data/shapenet_tmp/'
  this_target_n_faces = 1000
  dataset_name = 'partnet'
  fileds_needed = ['vertices', 'faces', 'edge_features', 'edges_map', 'edges', 'kdtree_query',
                   'label', 'labels', 'dataset_name', 'area_vertices_list']

  processed_md5 = [x.split('_')[0] for x in os.listdir(out_path)]
  if not os.path.isdir(out_path):
    os.makedirs(out_path)

  for d in tqdm(os.listdir(sn_chairs_path)):
    if d in processed_md5:
      continue
    if not os.path.isfile(sn_chairs_path + '/' + d + '/model.obj'):
      continue
    mesh_orig = load_meshes(sn_chairs_path + '/' + d + '/model.obj')
    # model_name, label, points10k, labels10k = get_partnet_labels(partnet_path + '/' + d)
    # mesh, labels, str_to_add = remesh(mesh_orig, this_target_n_faces, add_labels=True, labels_orig=labels_orig)
    mesh, labels, str_to_add = remesh(mesh_orig, this_target_n_faces)
    mesh_data = EasyDict(
      {'vertices': np.asarray(mesh.vertices), 'faces': np.asarray(mesh.triangles)})
    out_fc_full = out_path + d + str_to_add
    if os.path.isfile(out_fc_full + '.npz'):
      continue
    add_fields_and_dump_model(mesh_data, fileds_needed, out_fc_full, dataset_name)
    if 0:
      utils.visualize_model(mesh_data['vertices'], mesh_data['faces'],
                            cpos=[(-2., -0.2, 3.3), (0., -0.3, 0.1), (0., 1., 0.)])
    processed_md5.append(d)


def prepare_cubes(dataset_name='cubes', labels2use=cubes_labels,
                  path_in=os.path.expanduser('~') + '/datasets/cubes/',
                  p_out=os.path.expanduser('~') + '/runs/datasets_processed/cubes_tmp'):
  if not os.path.isdir(p_out):
    os.makedirs(p_out)

  for i, name in enumerate(labels2use):
    print('-->>>', name)
    for part in ['test', 'train']:
      pin = path_in + name + '/' + part + '/'
      prepare_directory_from_scratch(dataset_name, pathname_expansion=pin + '*.obj',
                                     p_out=p_out, add_labels=dataset_name, fn_prefix=part + '_', n_target_faces=[np.inf])


def prepare_human_body_segmentation(human_seg_path=os.path.expanduser('~') + '/datasets/sig17_seg_benchmark/',
                                    p_out=os.path.expanduser('~') + '/runs/datasets_processed/human_seg_tmp__'):
  def _fix_mesh(mesh, model_name, verbose=False):
    flip_orientation_fn = ['test/shrec/7.off', 'test/shrec/8.off', 'test/shrec/9.off', 'test/shrec/1.off',
                           'test/shrec/11.off', 'test/shrec/12_fix_orientation.off']
    if np.any([model_name.endswith(to_check) for to_check in flip_orientation_fn]):
      if verbose:
        print('Orientation changed')
      vertices = np.asarray(mesh.vertices)[:, [0, 2, 1]]
      mesh.vertices = open3d.utility.Vector3dVector(vertices)
    if model_name.find('/scape/') != -1:
      if verbose:
        print('Orientation changed 2')
      vertices = np.asarray(mesh.vertices)[:, [1, 0, 2]]
      mesh.vertices = open3d.utility.Vector3dVector(vertices)
    if model_name.endswith('test/shrec/12_fix_orientation.off'):
      if verbose:
        print('Scaling factor 10')
      vertices = np.asarray(mesh.vertices) / 10
      mesh.vertices = open3d.utility.Vector3dVector(vertices)
    if model_name.find('/adobe/') != -1:
      if verbose:
        print('Scaling factor 100')
      vertices = np.asarray(mesh.vertices) / 100
      mesh.vertices = open3d.utility.Vector3dVector(vertices)
    return mesh

  fileds_needed = ['vertices', 'faces', 'edge_features', 'edges_map', 'edges', 'kdtree_query',
                   'label', 'labels', 'dataset_name', 'area_vertices_list']
  fileds_needed += ['geo_dist']
  #fileds_needed += model_net['walk_cache', 'walk_cache_jumps']
  n_target_faces = [1000, 2000, 4000, 8000][:1]
  dataset_name = 'sig17_seg_benchmark'
  if not os.path.isdir(p_out):
    os.makedirs(p_out)
  for part in ['test', 'train']:
    print('part: ', part)
    path_meshes = human_seg_path + '/meshes/' + part
    seg_path = human_seg_path + '/segs/' + part
    all_fns = []
    for fn in Path(path_meshes).rglob('*.*'):
      all_fns.append(fn)
    #all_fns = np.random.permutation(all_fns)
    for fn in tqdm(all_fns):
      model_name = str(fn)
      #if model_namemodel_net.find('.ply') == -1:
      #  continue
      if model_name.endswith('.obj') or model_name.endswith('.off') or model_name.endswith('.ply'):
        new_fn = model_name[model_name.find(part) + len(part) + 1:]
        new_fn = new_fn.replace('/', '_')
        new_fn = new_fn.split('.')[-2]
        out_fn = p_out + '/' + part + '__' + new_fn
        mesh = load_mesh(model_name)
        mesh = mesh_orig = _fix_mesh(mesh, model_name)
        mesh_data = EasyDict({'vertices': np.asarray(mesh.vertices), 'faces': np.asarray(mesh.triangles)})
        face_labels = get_sig17_seg_bm_labels(mesh_data, model_name, seg_path)
        labels_orig = calc_vertex_labels_from_face_labels(mesh_data, face_labels)
        if 0:
          print(model_name)
          print('min: ', np.min(mesh_data['vertices'], axis=0))
          print('max: ', np.max(mesh_data['vertices'], axis=0))
          cpos = [(-3.5, -0.12, 6.0), (0., 0., 0.1), (0., 1., 0.)]
          utils.visualize_model(mesh_data['vertices'], mesh_data['faces'], vertex_colors_idx=labels_orig, cpos=cpos)
        add_labels = 1
        label = -1
        for this_target_n_faces in n_target_faces:
          mesh, labels, str_to_add = remesh(mesh_orig, this_target_n_faces, add_labels=add_labels, labels_orig=labels_orig)
          mesh_data = EasyDict({'vertices': np.asarray(mesh.vertices), 'faces': np.asarray(mesh.triangles), 'label': label, 'labels': labels})
          out_fc_full = out_fn + str_to_add
          if os.path.isfile(out_fc_full + '.npz'):
            continue
          add_fields_and_dump_model(mesh_data, fileds_needed, out_fc_full, dataset_name)
          if 0:
            utils.visualize_model(mesh_data['vertices'], mesh_data['faces'], vertex_colors_idx=mesh_data['labels'].astype(np.int),
                                  cpos=[(-2., -0.2, 3.3), (0., -0.3, 0.1), (0., 1., 0.)])


def prepare_human_seg_from_meshcnn(dataset_name='human_seg_from_meshcnn', labels2use=coseg_labels,
                  path_in=os.path.expanduser('~') + '/datasets/human_seg/',
                  p_out_root=os.path.expanduser('~') + '/runs/datasets_processed/human_seg_tmp_nrmls'):
  p_out = p_out_root + '/'

  for part in ['test', 'train']:
    pin = path_in + '/' + part + '/'
    prepare_directory_from_scratch(dataset_name, pathname_expansion=pin + '*.obj',
                                   p_out=p_out, add_labels=dataset_name, fn_prefix=part + '_', n_target_faces=[np.inf],
                                   extra_fields_needed=['geo_dist'], fix_mesh_fn=fix_mesh_human_body_from_meshcnn)

def prepare_coseg(dataset_name='coseg', labels2use=coseg_labels,
                  path_in=os.path.expanduser('~') + '/datasets/coseg/',
                  p_out_root=os.path.expanduser('~') + '/runs/datasets_processed/coseg_tmp2'):
  for sub_folder in os.listdir(path_in):
    p_out = p_out_root + '/' + sub_folder
    if not os.path.isdir(p_out):
      os.makedirs(p_out + '/' + sub_folder)

    for part in ['test', 'train']:
      pin = path_in + '/' + sub_folder + '/' + part + '/'
      prepare_directory_from_scratch(sub_folder, pathname_expansion=pin + '*.obj',
                                     p_out=p_out, add_labels=dataset_name, fn_prefix=part + '_', n_target_faces=[np.inf])

def map_fns_to_label(path=None, filenames=None):
  lmap = {}
  if path is not None:
    iterate = glob.glob(path + '/*.npz')
  elif filenames is not None:
    iterate = filenames

  for fn in iterate:
    mesh_data = np.load(fn, encoding='latin1', allow_pickle=True)
    label = int(mesh_data['label'])
    if label not in lmap.keys():
      lmap[label] = []
    if path is None:
      lmap[label].append(fn)
    else:
      lmap[label].append(fn.split('/')[-1])
  return lmap


def change_train_test_split(path, n_train_examples, n_test_examples, split_name):
  np.random.seed()
  fns_lbls_map = map_fns_to_label(path)
  for label, fns_ in fns_lbls_map.items():
    fns = np.random.permutation(fns_)
    assert len(fns) == n_train_examples + n_test_examples
    train_path = path + '/' + split_name + '/train'
    if not os.path.isdir(train_path):
      os.makedirs(train_path)
    test_path = path + '/' + split_name + '/test'
    if not os.path.isdir(test_path):
      os.makedirs(test_path)
    for i, fn in enumerate(fns):
      out_fn = fn.replace('train_', '').replace('test_', '')
      if i < n_train_examples:
        shutil.copy(path + '/' + fn, train_path + '/' + out_fn)
      else:
        shutil.copy(path + '/' + fn, test_path + '/' + out_fn)

def collect_n_models_per_class(in_path, n_models4train):
  all_files = os.listdir(in_path)
  all_files = [file for file in all_files if file.endswith('.npz') and file.find('train') != -1]
  for this_n_models in n_models4train:
    this_o_path = in_path + '/train_subset-tmp/' + str(this_n_models).zfill(3) + '_model_for_train'
    if not os.path.isdir(this_o_path):
      os.makedirs(this_o_path)
    for file in all_files[:this_n_models]:
      shutil.copy(in_path + '/' + file, this_o_path + '/' + file)

def prepare_shrec11_from_raw():
  # Prepare labels per model name
  current_label = None
  model_number2label = [-1 for _ in range(600)]
  for line in open(os.path.expanduser('~') + '/datasets/shrec11/evaluation/test.cla'):
    sp_line = line.split(' ')
    if len(sp_line) == 3:
      name = sp_line[0].replace('_test', '')
      if name in shrec11_labels:
        current_label = name
      else:
        raise Exception('?')
    if len(sp_line) == 1 and sp_line[0] != '\n':
      model_number2label[int(sp_line[0])] = shrec11_shape2label[current_label]


  # Prepare npz files
  p_in = os.path.expanduser('~') + '/datasets/shrec11/raw/'
  p_out = os.path.expanduser('~') + '/runs/datasets_processed/shrec11_raw_4k/'
  #prepare_directory_from_scratch('shrec11', pathname_expansion=p_in + '*.off',
  #                               p_out=p_out, add_labels=model_number2label, n_target_faces=[4000])

  # Prepare split train / test
  change_train_test_split(p_out, 16, 4, '16-04_C')


if __name__ == '__main__':
  TEST_FAST = 0
  # utils.config_gpu(False)
  np.random.seed(1)

  if len(sys.argv) > 1:
    if sys.argv[1] == 'modelnet40':
      prepare_modelnet40()
  elif 0: # test modelnet40
    prepare_modelnet40(part=0, size_limit=np.inf, fast_mode=False)
  elif 0: # FAUST
    if 1:
      pin = '/home/alon/datasets/MPI-FAUST/test/scans/'
      prepare_directory_from_scratch('faust_scans', pathname_expansion=pin +  '*.ply', p_out='/home/alon/runs/cache/faust_scans_tmp2/', add_labels=False)
    else:
      prepare_directory_from_scratch('faust')
  elif 0: # PartNet
    prepare_part_net_using_points()
  elif 1:  # PartNet
    prepare_shapenet_chairs_using_mesh()
  elif 0:
    #prepare_human_body_segmentation()
    prepare_human_seg_from_meshcnn()
  elif 0:
    prepare_cubes()
  elif 0:
    prepare_cubes(dataset_name='shrec11', path_in=os.path.expanduser('~') + '/datasets/shrec_16/',
                  p_out=os.path.expanduser('~') + '/runs/datasets_processed/shrec11_tmp',
                  labels2use=shrec11_labels)
  elif 0:
    prepare_coseg()
  elif 0:
    change_train_test_split(path=os.path.expanduser('~') + '/runs/datasets_processed/shrec11/',
                            n_train_examples=16, n_test_examples=4, split_name='16-04_C')
  elif 0:
    collect_n_models_per_class(in_path=os.path.expanduser('~') + '/runs/datasets_processed/coseg/coseg_vases/',
                               n_models4train=[1, 2, 4, 8, 16, 32])
