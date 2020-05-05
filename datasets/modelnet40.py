import os
import torch
from torch.utils.data import Dataset, DataLoader
from utils.mesh import load_npz
from utils.walks import generate_walk
import numpy as np

class ModelNet40(Dataset):
  def __init__(self, params):
    super(ModelNet40, self).__init__()
    self.label_to_names = {0: 'airplane',
                           1: 'bathtub',
                           2: 'bed',
                           3: 'bench',
                           4: 'bookshelf',
                           5: 'bottle',
                           6: 'bowl',
                           7: 'car',
                           8: 'chair',
                           9: 'cone',
                           10: 'cup',
                           11: 'curtain',
                           12: 'desk',
                           13: 'door',
                           14: 'dresser',
                           15: 'flower_pot',
                           16: 'glass_box',
                           17: 'guitar',
                           18: 'keyboard',
                           19: 'lamp',
                           20: 'laptop',
                           21: 'mantel',
                           22: 'monitor',
                           23: 'night_stand',
                           24: 'person',
                           25: 'piano',
                           26: 'plant',
                           27: 'radio',
                           28: 'range_hood',
                           29: 'sink',
                           30: 'sofa',
                           31: 'stairs',
                           32: 'stool',
                           33: 'table',
                           34: 'tent',
                           35: 'toilet',
                           36: 'tv_stand',
                           37: 'vase',
                           38: 'wardrobe',
                           39: 'xbox'}

    self.params = params
    self.npz_path = './Data/modelnet40_1k2k4k'
    # TODO: need to check if we can load npz in __getitem__ instead of before,
    #  initial loading takes long time, but might save time in train loops
    self.input_meshes = {'train': [], 'val': [], 'test': [], 'all': {}}
    self._get_splits()


    self.mode = 'train'

  def _get_splits(self):
    '''
    Loading train/val/test into placeholders
    :return:
    '''
    for split in ['train', 'test']:
      cur_npz_path = os.path.join(self.npz_path, split + '*.npz')
      cur_filenames = self._get_filenames(cur_npz_path)
      #TODO: load before this line
      for fn in cur_filenames:
        mesh_data = fn
        self.input_meshes[split].append(mesh_data)

  def _get_filenames(self, path):
    import glob
    filenames_ = glob.glob(path)
    filenames = []
    for fn in filenames_:
      try:
        n_faces = int(fn.split('.')[-2].split('_')[-1])
        if n_faces > self.params.train_min_max_faces2use[1] or n_faces < self.params.train_min_max_faces2use[0]:
          continue
      except:
        pass
      filenames.append(fn)
    assert len(filenames) > 0,\
      'DATASET error: no files in directory to be used! \nDataset directory: ' + path

    return filenames

  def _get_walks(self, mesh):
    walks, _ = generate_walk(mesh, self.params)
    return walks

  def __len__(self):
    if self.mode == 'test':
      return len(self.input_meshes[self.mode]) * 32
    return len(self.input_meshes[self.mode])

  def __getitem__(self, item):
    if self.mode == 'test':
        item = item // 32
    mesh = load_npz(self.input_meshes[self.mode][item], self.params)
    walks = self._get_walks(mesh)
    return mesh, walks

  def collate_fn(self, batch):
    returndict = {}
    returndict['mesh_data'] = [x[0] for x in batch]
    returndict['walks'] = torch.cat([torch.from_numpy(x[1]) for x in batch])
    returndict['label'] = torch.stack([torch.from_numpy(x[0]['labels']) for x in batch])
    return returndict

  def test(self):
    self.mode = 'test'

  def train(self):
    self.mode = 'train'

  def val(self):
    self.mode = 'val'

