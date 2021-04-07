import random
import torch
import os
from pathlib import Path
import numpy as np


lr_calc = lambda batch_size: 0.128 / 2048 * batch_size


target_cols = [
    'ETT - Abnormal',
    'ETT - Borderline',
    'ETT - Normal',
    'NGT - Abnormal',
    'NGT - Borderline',
    'NGT - Incompletely Imaged',
    'NGT - Normal',
    'CVC - Abnormal',
    'CVC - Borderline',
    'CVC - Normal',
    'Swan Ganz Catheter Present'
]


class config:
    experiment_prefix = ''
    n_workers = 8
    num_classes = 11
    seed = 42
    image_size = 1000
    annot_size = 50
    model_name = 'tf_efficientnet_b7_ns'
    train_batch_size = 8
    valid_batch_size = 32
    lr = lr_calc(train_batch_size)
    min_lr = 1e-6
    experiment_name = f'{experiment_prefix}{model_name}_{image_size}'
    device = 'cuda'
    devices = (0, 1, 2, 3)


def get_best_model_path(folder_path):
    model_path = Path(folder_path).glob('*.pth')
    model_path = sorted(model_path)[-1]
    print(model_path)
    return model_path


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
