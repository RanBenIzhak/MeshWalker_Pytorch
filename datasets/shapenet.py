#############################################
#   ShapeNet for MeshWalk Pytorch Dataset   #
#############################################

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import os
import numpy as np
from glob import glob
from utils.mesh import load_npz
from utils.walks import generate_walk

class ExactMatchChairs(Dataset):
  def __init__(self, path, params):
    super(ExactMatchChairs, self).__init__()
    self.parmas=params
    self.path=path
    # We use ExactMatch (Li et al. 2015 - Joint embedding) chairs split
    self._get_splits()
    self.images = {'train': [], 'val': [], 'test': []}
    self.imid2shapeid = {'train': [], 'val': [], 'test': []}
    self.file = {'train': [], 'val': [], 'test': []}
    self.num_examples = {'training': [], 'validation': [], 'test': []}
    self.input_labels = {'training': [], 'validation': [], 'test': []}
    self.shape_feats = {}
    self.input_meshes = {'training': [], 'validation': [], 'test': [], 'all': {}}

  def _get_splits(self):
    # ==== Constants, paths, loading files needed for data management
    data_folder = '/home/ran/PycharmProjects/JointEmbedding-master/data'
    base_filename = os.path.join(data_folder, 'image_embedding', 'syn_images_{}_03001627_{}.txt')
    shapelist_filename = 'shape_list_03001627.txt'  # 0 to 6777  (NOT same as in ExactMatch folder!)
    self.shapeid2shapemd = [x[:-1].split(' ')[1] for x in
                            open(os.path.join(data_folder, shapelist_filename)).readlines()]  # 0 to 6777

    # ============= Loading Images list for all sets ==========
    # === train/val ===
    for m in ['training', 'validation']:
      self.file[m] = base_filename.format('filelist', m)
      self.imid2shapeid[m] = [int(x[:-1]) for x in open(base_filename.format('imageid2shapeid', m)).readlines()]
      self.images[m] = open(self.file[m]).read().splitlines()
      assert len(self.imid2shapeid[m]) == len(self.images[m])
    # === test ===
    self.file[
      'test'] = '/home/ran/PycharmProjects/JointEmbedding-master/src/experiments/ExactMatchChairsDataset/exact_match_chairs_img_filelist.txt'
    self.images['test'] = [x[:-1] for x in open(self.file['test']).readlines()]
    self.images['test'] = [x.replace('/orions3-zfs/projects/rqi/Dataset',
                                     '/home/ran/PycharmProjects/JointEmbedding-master/src/experiments') for x in
                           self.images['test']]
    test_shapemd_list = [x.split('/')[-1].split('_')[0] for x in self.images['test']]
    self.imid2shapeid['test'] = [self.shapeid2shapemd.index(md5) for md5 in test_shapemd_list]

    # =============== Guibas pre-computed features ===================== #
    ses = os.path.join(data_folder, 'shape_embedding', 'shape_embedding_space_03001627.txt')
    self.pc_guibas_embedding = open(ses).readlines()
    self.pc_guibas_embedding = [x[:-1].split(' ') for x in self.pc_guibas_embedding]
    self.pc_guibas_embedding = np.asarray([[float(x) for x in y] for y in self.pc_guibas_embedding])

    # =============== Excluding all images corresponding to shapes in the test set ==================
    # sort out test models/images from train set.
    # test indexing is 0 to 6776, train/val indexing is 0 to 6777 (using md5 to overcome this)
    exactmatch_md5_file = '/home/ran/PycharmProjects/JointEmbedding-master/src/experiments/ExactMatchChairsDataset/filelist_exactmatch_chair_105.txt'
    self.exclude_md5 = open(exactmatch_md5_file).read().splitlines()
    exclude_indices = [i for i, x in enumerate(self.images['train']) if
                       self.shapeid2shapemd[self.imid2shapeid['train'][i]] in self.exclude_md5]
    for i in exclude_indices[::-1]:  # need to delete from end to beginning
      del self.images['train'][i]
      del self.imid2shapeid['train'][i]
    # Testing that we have no matches left
    exclude_indices = [i for i, x in enumerate(self.images['train']) if
                       self.shapeid2shapemd[self.imid2shapeid['train'][i]] in self.exclude_md5]
    assert not exclude_indices

    # =============== Setting npz paths =========== #
    self.npz_path = '/home/ran/PycharmProjects/MeshWalker_Pytorch/Data/shapenet_tmp'

    for mode in ['training', 'validation', 'test']:
      self.num_examples[mode] = len(self.images[mode])
      self.input_labels[mode] = np.array([2 for _ in range(self.num_examples[mode])])

    for md5 in self.shapeid2shapemd:
      npz_path = glob(os.path.join(self.npz_path, md5 + '*.npz'))
      mesh = load_npz(npz_path)
      self.input_meshes['all'][md5] = mesh

  def __len__(self):
    return len(self.images[self.mode])

  def __getitem__(self, item):
    # TODO: IMPORTANT!!! ADD EXCLUDE IDS OF TEST SET SHAPES!! SKIP/IGNORE THEM
    # TESTING
    if not self.use_features:  # or self.mode == 'test':
      tfs = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225]),
                                ])
      image = Image.open(self.images[self.mode][item]).convert("RGB")
      # image = resize_pad(image)
      image = image.resize((227, 227))
      image_tensor = tfs(image)
      image_input = image_tensor
    else:
      image_input = torch.from_numpy(np.asarray(self.features[self.mode].get(str(item))))

    if self.mode == 'test':  # ExactMatch databases testing
      im2shape_md5 = self.images[self.mode][item].split('/')[-1].split('_')[0]
    else:
      im2shape_md5 = self.images[self.mode][item].split('/')[-2]
    mesh = self.input_meshes[im2shape_md5]
    walks = self._get_walks(mesh)
    # TODO: return projected locations of shape in image

    # for more details, return shape info... (or not)
    return image_input, walks, im2shape_md5, 2


  def _get_walks(self, mesh):
    # first we need to get some params
    walks, _ = generate_walk(mesh, self.params)

  def test(self):
    self.mode = 'test'

  def train(self):
    self.mode = 'train'

  def val(self):
    self.mode = 'val'

  @staticmethod
  def collate_fn(batch):
    returndict = {}
    returndict['image'] = torch.stack([x[0] for x in batch])
    returndict['walks'] = torch.stack([x[1] for x in batch])
    returndict['instance_label'] = returndict['label'] = [x[2] for x in batch]
    return returndict
