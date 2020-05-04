from easydict import EasyDict
import numpy as np
import os
import open3d
import torch
import time
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from utils.utils import RunningWindow
from utils.logging import backup_python_files_and_params

# If we run debug mode - to change to false later
DEBUG=True

LOAD_NET = True

# =========== Paramaters ========= #
from utils.parameters import modelnet_params

params = modelnet_params()

# ========== Set logging folder, save run parameters ========= #
backup_python_files_and_params(params)

# ========== Set Database and Dataloader ========= #
from datasets.modelnet40 import ModelNet40
dataset = ModelNet40(params)
train_dl = DataLoader(dataset, batch_size=params.batch_size, shuffle=True, num_workers=4,
                      collate_fn=dataset.collate_fn, drop_last=False)
test_dl = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                      collate_fn=dataset.collate_fn)

# ===== Set model ===== #
#currently only RnnWalkNet available - MeshWalker
from models.MeshWalker import RnnWalkNet
dnn_model = RnnWalkNet(params).cuda()
loss_fn = torch.nn.CrossEntropyLoss().cuda()

if LOAD_NET:
  dnn_model.load_state_dict(torch.load('/home/ran/PycharmProjects/MeshWalker_Pytorch/weights/modelnet40_best.pth.tar'))
# ====  set optimizer ==== #
if params.optimizer_type == 'adam':
  optimizer = torch.optim.Adam(params=dnn_model.parameters(), lr=params.learning_rate[0])  #lr=params.learning_rate[0])
  clipnorm = params.gradient_clip_th
elif params.optimizer_type == 'cycle':
  # TODO: test cylic LR in pytorch
  optimizer = torch.optim.SGD(dnn_model.parameters(), params.learning_rate[0], momentum=0.9)
  scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
                                                base_lr=params.cycle_opt_prms.initial_learning_rate,
                                                max_lr=params.cycle_opt_prms.maximal_learning_rate,
                                                step_size_up=params.cycle_opt_prms.step_size)
elif params.optimizer_type == 'sgd':
  optimizer = torch.optim.SGD(dnn_model.parameters(), lr=params.learning_rate[0], decay=0, momentum=0.9, nesterov=True, clipnorm=params.gradient_clip_th)
else:
  raise Exception('optimizer_type not supported: ' + params.optimizer_type)

# ==== Time stamps messages, loggers etc. ==== #
time_msrs = {x: 0 for x in ['train_step', 'train_step_triplet', 'get_train_data', 'test']}



# ====== Train loop ====== #
cur_lr = params.learning_rate[0]
best_val_acc = 0
cur_loss = RunningWindow(N=1000)

for epoch in range(params.EPOCHS):
  dataset.train()
  dnn_model.train()
  epoch_loss = 0
  epoch_acc = 0
  if epoch in [50, 100, 150]:
    for param_group in optimizer.param_groups:
      cur_lr = cur_lr * 0.7
      param_group['lr'] = cur_lr
      print('Reduced LR to {}'.format(cur_lr))
  for i, batch in enumerate(train_dl):
    batch_start_time = time.time()
    output = dnn_model(batch['walks'].cuda())
    net_out_time = time.time() - batch_start_time
    # print('Batch output in net - {:2.3f} seconds'.format(net_out_time))
    labels = batch['label'].cuda()
    ce_loss = loss_fn(output[1], labels)

    # correct % in batch
    pred_choice = output[1].data.max(1)[1]
    correct = pred_choice.eq(labels.data).cpu().sum()

    # Regularization loss
    l2_reg = None
    for W in dnn_model.parameters():
      if l2_reg is None:
        l2_reg = W.norm(2)
      else:
        l2_reg = l2_reg + W.norm(2)
    loss = ce_loss + 0.0001 * l2_reg

    # TODO: log loss value
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(dnn_model.parameters(), params.gradient_clip_th)
    optimizer.step()
    if params.optimizer_type == 'cycle':
      scheduler.step()
    # print('Update time - {:2.3f} seconds'.format(time.time() - net_out_time - batch_start_time))
    cur_loss.update(ce_loss.item())
    epoch_loss += ce_loss.item()
    epoch_acc += correct.float()
    if i % 30 == 0:
      print('Cur loss: {:2.3f}, batch accuracy: {:2.3f}'.format(cur_loss(), correct.float() / params.batch_size))

  print('Epoch {} Loss: {:2.3f}\t Accuracy: {:2.3f}'.format(epoch,
                                                           epoch_loss / (i+1),
                                                           epoch_acc / ((i+1) * params.batch_size)))
  if (epoch+1) % 5 == 0:
    dnn_model.eval()
    dataset.test()
    epoch_outputs = []
    labels_outputs = []
    with torch.no_grad():
      for i, batch in enumerate(test_dl):
        output = dnn_model(batch['walks'].cuda())
        labels = batch['label'].cuda()

        epoch_outputs.append(output[1].data.mean(dim=0).argmax())
        labels_outputs.append(labels)
        if not ((epoch + 1) % 20 == 0) and i == 120:
          print('Short test run')
          break
    preds = torch.stack(epoch_outputs).cpu().numpy()
    targets = torch.cat(labels_outputs).cpu().numpy()
    test_acc = np.float(np.sum(preds == targets)) / len(targets)
    conf_mat = confusion_matrix(targets, preds)
    print('Epoch {} test accuracy: {:2.3f}'.format(epoch, test_acc))

    if test_acc > best_val_acc:
      best_val_acc = test_acc
      save_dir = os.path.join(params.logdir, 'weights')
      if not os.path.exists(save_dir):
        os.makedirs(save_dir)
      torch.save(dnn_model.state_dict(), os.path.join(save_dir, 'modelnet40_best.pth.tar'))



