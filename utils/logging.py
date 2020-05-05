import os
import shutil
import json
from copy import copy
import numpy as np
import datetime
import logging


def set_logger(log_path):
  """Set the logger to log info in terminal and file `log_path`.
  In general, it is useful to have a logger so that every output to the terminal is saved
  in a permanent file. Here we save it to `model_dir/train.log`.
  Example:
  ```
  logging.info("Starting training...")
  ```
  Args:
      log_path: (string) where to log
  """
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)

  if logger.handlers:
    logger.handlers = []
  # Logging to a file
  file_handler = logging.FileHandler(log_path)
  file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
  logger.addHandler(file_handler)

  # Logging to console
  stream_handler = logging.StreamHandler()
  stream_handler.setFormatter(logging.Formatter('%(message)s'))
  logger.addHandler(stream_handler)


def backup_python_files_and_params(params):
  save_id = 0
  while 1:
    code_log_folder = params.logdir + '/.' + str(save_id)
    if not os.path.isdir(code_log_folder):
      os.makedirs(code_log_folder)
      for file in os.listdir():
        if file.endswith('py'):
          shutil.copyfile(file, code_log_folder + '/' + file)
      break
    else:
      save_id += 1

  # Dump params to text file
  try:
    prm2dump = copy.deepcopy(params)
    if 'hyper_params' in prm2dump.keys():
      prm2dump.hyper_params = str(prm2dump.hyper_params)
      prm2dump.hparams_metrics = prm2dump.hparams_metrics[0]._display_name
      for l in prm2dump.net:
        l['layer_function'] = 'layer_function'
    with open(params.logdir + '/params.txt', 'w') as fp:
      json.dump(prm2dump, fp, indent=2, sort_keys=True)
  except:
    pass


def get_run_folder(root_dir, str2add='', cont_run_number=False):
  try:
    all_runs = os.listdir(root_dir)
    run_ids = [int(d.split('-')[0]) for d in all_runs if '-' in d]
    if cont_run_number:
      n = [i for i, m in enumerate(run_ids) if m == cont_run_number][0]
      run_dir = root_dir + all_runs[n]
      print('Continue to run at:', run_dir)
      return run_dir
    n = np.sort(run_ids)[-1]
  except:
    n = 0
  now = datetime.datetime.now()
  return root_dir + str(n + 1).zfill(4) + '-' + now.strftime("%d.%m.%Y..%H.%M") + str2add


def load_params(logdir):
  with open(logdir + '/params.txt', 'w') as fp:
    params = json.load(fp)
  return params
