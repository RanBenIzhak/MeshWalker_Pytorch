import os, shutil, json, copy
import time
from easydict import EasyDict

import numpy as np
import pylab as plt

import tensorflow as tf
import tensorflow_addons as tfa

import dnn_cad_seq
import dataset_directory
import dataset
import utils
import params_setting_2
import triplets_utils

print('tf.__version__', tf.__version__)

np.set_printoptions(suppress=True)

glb = EasyDict()
glb.gradients_segm = None
save_conf_matrix = False


def train_val(params, hyper_params_tuning=False):
  print(utils.color.BOLD + utils.color.RED + 'params.logdir :::: ', params.logdir, utils.color.END)
  print(utils.color.BOLD + utils.color.RED, os.getpid(), utils.color.END)
  utils.backup_python_files_and_params(params)

  train_datasets = []
  train_ds_iters = []
  max_train_size = 0
  for i in range(len(params.datasets2use['train'])):
    this_train_dataset, n_trn_items = dataset_directory.tf_mesh_dataset(params, params.datasets2use['train'][i],
                                                                   mode=params.network_tasks[i], size_limit=params.train_dataset_size_limit,
                                                                   shuffle_size=100, min_max_faces2use=params.train_min_max_faces2use,
                                                                   max_size_per_class=params.train_max_size_per_class, min_dataset_size=128,
                                                                   data_augmentation=params.train_data_augmentation)
    print('Train Dataset size:', n_trn_items)
    train_ds_iters.append(iter(this_train_dataset.repeat()))
    train_datasets.append(this_train_dataset)
    max_train_size = max(max_train_size, n_trn_items)
  train_epoch_size = max(8, int(max_train_size / params.n_walks_per_model / params.batch_size))
  print('train_epoch_size:', train_epoch_size)
  if params.datasets2use['test'] is None:
    test_dataset = None
    n_tst_items = 0
  else:
    test_dataset, n_tst_items = dataset_directory.tf_mesh_dataset(params, params.datasets2use['test'][0],
                                                                  mode=params.network_tasks[0], size_limit=params.test_dataset_size_limit,
                                                                  shuffle_size=100, min_max_faces2use=params.test_min_max_faces2use)
  print(' Test Dataset size:', n_tst_items)

  if params.net_start_from_prev_net is not None:
    init_net_using = params.net_start_from_prev_net
  else:
    #init_net_using = params.logdir + '/learned_model.keras'
    init_net_using = None

  if params.optimizer_type == 'adam':
    optimizer = tf.keras.optimizers.Adam(lr=params.learning_rate[0], clipnorm=params.gradient_clip_th)
  elif params.optimizer_type == 'cycle':
    lr_schedule = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=params.cycle_opt_prms.initial_learning_rate,
                                                      maximal_learning_rate=params.cycle_opt_prms.maximal_learning_rate,
                                                      step_size=params.cycle_opt_prms.step_size,
                                                      scale_fn=lambda x: 1., scale_mode="cycle", name="MyCyclicScheduler")
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=params.gradient_clip_th)
  elif params.optimizer_type == 'sgd':
    optimizer = tf.keras.optimizers.SGD(lr=params.learning_rate[0], decay=0, momentum=0.9, nesterov=True, clipnorm=params.gradient_clip_th)
  else:
    raise Exception('optimizer_type not supported: ' + params.optimizer_type)

  if params.net == 'RnnWalkNet':
    dnn_model = dnn_cad_seq.RnnWalkNet(params, params.n_classes, params.net_input_dim, init_net_using, optimizer=optimizer)
  elif params.net == 'Attention':
    dnn_model = dnn_cad_seq.AttentionWalkNet(params, params.n_classes, params.net_input_dim, init_net_using, optimizer=optimizer)
  elif params.net == 'RnnStride':
    dnn_model = dnn_cad_seq.RnnStrideWalkNet(params, params.n_classes, params.net_input_dim, params.net_start_from_prev_net, optimizer=optimizer)

  time_msrs = {}
  time_msrs_names = ['train_step', 'train_step_triplet', 'get_train_data', 'test']
  for name in time_msrs_names:
    time_msrs[name] = 0
  seg_train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='seg_train_accuracy')

  train_log_names = ['triplet_loss', 'scattered_from_one', 'seg_loss', 'triplet_neg', 'triplet_pos']
  train_logs = {name: tf.keras.metrics.Mean(name=name) for name in train_log_names}
  train_logs['seg_train_accuracy'] = seg_train_accuracy

  test_log_names = ['triplet_loss', 'feature_energy', 'pos_diff', 'neg_diff', 'rank', 'rank_shrec', 'mean_ngd', 'rnn_memory_indicator',
                    'successes_ratio', 'successes_ratio_with_margin']
  test_logs = {name:tf.keras.metrics.Mean(name=name) for name in test_log_names}

  # Classification train / test functions
  # -------------------------------------
  seg_loss = tf.keras.losses.SparseCategoricalCrossentropy()
  @tf.function
  def train_step(model_ftrs_, labels_, one_label_per_model):
    sp = model_ftrs_.shape
    model_ftrs = tf.reshape(model_ftrs_, (-1, sp[-2], sp[-1]))
    with tf.GradientTape() as tape:
      if one_label_per_model:
        labels = tf.reshape(tf.transpose(tf.stack((labels_,)*params.n_walks_per_model)),(-1,))
        predictions = dnn_model(model_ftrs)
      else:
        labels = tf.reshape(labels_, (-1, sp[-2]))
        skip = params.min_seq_len
        predictions = dnn_model(model_ftrs)[:, skip:]
        labels = labels[:, skip + 1:]
      seg_train_accuracy(labels, predictions)
      loss = seg_loss(labels, predictions)
      loss += tf.reduce_sum(dnn_model.losses)

    glb.gradients_segm = tape.gradient(loss, dnn_model.trainable_variables)
    optimizer.apply_gradients(zip(glb.gradients_segm, dnn_model.trainable_variables))

    train_logs['seg_loss'](loss)

    return loss

  test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
  @tf.function
  def test_step(model_ftrs_, labels_, one_label_per_model):
    sp = model_ftrs_.shape
    model_ftrs = tf.reshape(model_ftrs_, (-1, sp[-2], sp[-1]))
    if one_label_per_model:
      labels = tf.reshape(tf.transpose(tf.stack((labels_,) * params.n_walks_per_model)), (-1,))
      predictions = dnn_model(model_ftrs, training=False)
    else:
      labels = tf.reshape(labels_, (-1, sp[-2]))
      skip = params.min_seq_len
      predictions = dnn_model(model_ftrs, training=False)[:, skip:]
      labels = labels[:, skip + 1:]
    best_pred = tf.math.argmax(predictions, axis=-1)
    test_accuracy(labels, predictions)
    confusion = tf.math.confusion_matrix(labels=tf.reshape(labels, (-1,)), predictions=tf.reshape(best_pred, (-1,)),
                                         num_classes=params.n_classes)
    return confusion
  # -------------------------------------

  # Triplet-Loss train / test functions
  # -----------------------------------
  #@tf.function
  def train_step_triplet(model_ftrs_, index_labels_):
    # Assumptions:
    # - One model per batch (to fix later)
    # - 1st walk is the "negative" walk, 3 other are "positive" walks and candidates for positive examples
    # - Negative walk don't overlap positive ones
    # - Positive walks are not exactly the same, but some indices do overlap
    model_ftrs = model_ftrs_[0]
    index_labels = index_labels_[0, :, 1:]
    with tf.GradientTape() as tape:
      skip = params.min_seq_len
      embeddings = dnn_model(model_ftrs, classify=False)
      labels_, embeddings_ = index_labels[:, skip:], embeddings[:, skip:]
      embeddings = tf.reshape(embeddings_, (-1, embeddings.shape[-1]))
      if 1:
        n_pos_walks = int(params.n_walks_per_model / 2)
        labels4triplet = tf.reshape(tf.concat((tf.zeros_like(labels_[:n_pos_walks]), tf.ones_like(labels_[n_pos_walks:])), axis=0), [-1]) # 1st walk is far from the last ones
        loss = tfa.losses.triplet_semihard_loss(labels4triplet, embeddings)
        if 1:
          n_steps2use = embeddings_.shape[1]
          pairwise_dist = triplets_utils._pairwise_distances(embeddings)
          pos1 = pairwise_dist[:n_steps2use, :n_steps2use]
          pos2 = pairwise_dist[n_steps2use:, n_steps2use:]
          neg = pairwise_dist[:n_steps2use, n_steps2use:]
          pos_dist_mean = (tf.reduce_mean(pos1) + tf.reduce_mean(pos2)) / 2
          neg_dist_mean = tf.reduce_mean(neg)
      else:
        labels = tf.reshape(labels_, [-1])
        #labels_neg_walks = -1 * tf.ones_like(labels_[:1]) # Set the 1st walk to one unique label, so we will use it and only it only as negative example
        triplet_loss, fraction_positive_triplets, pos_dist_mean, neg_dist_mean = triplets_utils.batch_all_triplet_loss(labels, embeddings)
        loss = triplet_loss
      loss += tf.reduce_sum(dnn_model.losses)

    glb.gradients_segm = tape.gradient(loss, dnn_model.trainable_variables)
    optimizer.apply_gradients(zip(glb.gradients_segm, dnn_model.trainable_variables))

    train_logs['triplet_loss'](loss)
    train_logs['triplet_pos'](pos_dist_mean)
    train_logs['triplet_neg'](neg_dist_mean)

    return loss

  '''test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
  #@tf.function
  def test_step_triplet(model_ftrs_, labels_):
    sp = model_ftrs_.shape
    model_ftrs = tf.reshape(model_ftrs_, (-1, sp[-2], sp[-1]))
    if labels_.ndim == 1:
      labels = tf.reshape(tf.transpose(tf.stack((labels_,) * params.n_walks_per_model)), (-1,))
      predictions = dnn_model(model_ftrs, training=False).numpy()
    else:
      labels = tf.reshape(labels_, (-1, sp[-2]))
      skip = params.min_seq_len
      predictions = dnn_model(model_ftrs, training=False).numpy()[:, skip:]
      labels = labels[:, skip + 1:]
    best_pred = np.argmax(predictions, axis=-1)
    test_accuracy(labels, predictions)
    confusion = tf.math.confusion_matrix(labels=labels.numpy().flatten(), predictions=best_pred.flatten(),
                                         num_classes=params.n_classes).numpy()
    return confusion'''
  # -------------------------------------

  one_label_per_model = params.network_task == 'classification'
  next_iter_to_log = 0
  e_time = 0
  accrcy_smoothed = tb_epoch = last_loss = all_confusion = None
  all_confusion = {}
  with tf.summary.create_file_writer(params.logdir).as_default():
    #tf.summary.trace_on(graph=False, profiler=True)
    epoch = 0
    while epoch < params.EPOCHS:
      epoch += 1
      str_to_print = str(os.getpid()) + ') Epoch' + str(epoch) + ', iter ' + str(optimizer.iterations.numpy())

      # Save some logs & infos
      utils.save_model_if_needed(optimizer.iterations, dnn_model, params)
      if tb_epoch is not None:
        e_time = time.time() - tb_epoch
        tf.summary.scalar('time/one_epoch', e_time, step=optimizer.iterations)
        tf.summary.scalar('time/av_one_trn_itr', e_time / n_iters, step=optimizer.iterations)
        for name in time_msrs_names:
          if time_msrs[name]:  # if there is something to save
            tf.summary.scalar('time/' + name, time_msrs[name], step=optimizer.iterations)
            time_msrs[name] = 0
      tb_epoch = time.time()
      n_iters = 0
      tf.summary.scalar(name="train/learning_rate", data=optimizer._decayed_lr(tf.float32), step=optimizer.iterations)
      tf.summary.scalar(name="mem/free", data=utils.check_mem_and_exit_if_full(), step=optimizer.iterations)

      # Train one EPOC
      str_to_print += '; LR: ' + str(optimizer._decayed_lr(tf.float32))
      train_logs['seg_loss'].reset_states()
      tb = time.time()
      for iter_db in range(train_epoch_size):
        for dataset_id in range(len(train_datasets)):
          name, model_ftrs, labels = train_ds_iters[dataset_id].next()
          dataset_type = utils.get_dataset_type_from_name(name)
          if params.learning_rate_dynamics != 'stable':
            utils.update_lerning_rate_in_optimizer(0, params.learning_rate_dynamics, optimizer, params)
          time_msrs['get_train_data'] += time.time() - tb
          n_iters += 1
          tb = time.time()
          if params.train_loss[dataset_id] == 'cros_entr':
            train_step(model_ftrs, labels, one_label_per_model=one_label_per_model)
            loss2show = 'seg_loss'
          elif params.train_loss[dataset_id] == 'triplet':
            train_step_triplet(model_ftrs, labels)
            loss2show = 'triplet_loss'
          else:
            raise Exception('Unsupported loss_type: ' + params.train_loss[dataset_id])
          time_msrs['train_step'] += time.time() - tb
          tb = time.time()
        if iter_db == train_epoch_size - 1:
          str_to_print += ', TrnLoss: ' + str(round(train_logs[loss2show].result().numpy(), 2))

      if optimizer.iterations >= next_iter_to_log:
        utils.log_gradients(dnn_model, glb.gradients_segm, optimizer.iterations)
        for k, v in train_logs.items():
          if v.count.numpy() > 0:
            tf.summary.scalar('train/' + k, v.result(), step=optimizer.iterations)
            v.reset_states()
        next_iter_to_log += params.log_freq

      if test_dataset is not None:
        n_test_iters = 0
        tb = time.time()
        for name, model_ftrs, labels in test_dataset:
          n_test_iters += model_ftrs.shape[0]
          if n_test_iters > params.n_models_per_test_epoch:
            break
          confusion = test_step(model_ftrs, labels, one_label_per_model=one_label_per_model)
          dataset_type = utils.get_dataset_type_from_name(name)
          if dataset_type in all_confusion.keys():
            all_confusion[dataset_type] += confusion
          else:
            all_confusion[dataset_type] = confusion
        if accrcy_smoothed is None:
          accrcy_smoothed = test_accuracy.result()
        accrcy_smoothed = accrcy_smoothed * .9 + test_accuracy.result() * 0.1
        tf.summary.scalar('test/accuracy_' + dataset_type, test_accuracy.result(), step=optimizer.iterations)
        tf.summary.scalar('test/accuracy_smoothed', accrcy_smoothed, step=optimizer.iterations)
        str_to_print += ', test/accuracy_' + dataset_type + ': ' + str(round(test_accuracy.result().numpy(), 2))
        test_accuracy.reset_states()
        if save_conf_matrix:
          for conf_name, conf_mat in all_confusion.items():
            if conf_mat.sum() > 1000:
              conf_mat = conf_mat / (np.sum(conf_mat, axis=1) + 1e-6)
              conf_mat = utils.colorize(conf_mat)
              tf.summary.image('conf_mat/' + conf_name, conf_mat[None, :, :, :], step=optimizer.iterations)
              conf_mat = None
        time_msrs['test'] += time.time() - tb

      str_to_print += ', time: ' + str(round(e_time, 1))
      print(str_to_print) 

      if tf.equal(optimizer.iterations % params.log_freq, 0):
        for log_name, log_list in zip(['test'],
                                      [test_logs]):
          for v in log_list.values():
            if v.count.numpy(): # if there is something to save
              tf.summary.scalar(log_name + '/' + v.name, v.result(), step=optimizer.iterations)
              v.reset_states()

      train_logs['triplet_loss'].reset_states()
      #tf.summary.trace_export('trace_export', profiler_outdir=params.logdir, step=optimizer.iterations)

  return last_loss

if __name__ == '__main__':
  np.random.seed(0)
  utils.config_gpu()
  # Classifications
  if 1:
    params = params_setting_2.modelnet_params()

  if 0:
    params = params_setting_2.shrec11_params()

  if 0:
    params = params_setting_2.cubes_params()

  # Semantic Segmentations
  if 0:
    params = params_setting_2.human_seg_params()

  if 0:
    params = params_setting_2.coseg_params('aliens')
    #params = params_setting_2.coseg_params('chairs')
    #params = params_setting_2.coseg_params('vases')

  if 0:
    params = params_setting_2.dancer_params()

  if 0:
    params = params_setting_2.faust_params()

  # Transfer-learning / Semisupervised / Unsupervised
  if 0:
    params = params_setting_2.unsupervised_human_seg_params()

  if 0:
    params = params_setting_2.semi_supervised_human_seg_params()

  if 0:
    params = params_setting_2.transfer_learning_params()


  train_val(params)
