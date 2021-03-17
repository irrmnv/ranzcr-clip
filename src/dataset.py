import albumentations as A
from torch.utils.data import Dataset
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .commons import config, target_cols



def get_train_data():
    train = pd.read_csv('/workspace/train.csv')

    train_annotated = pd.read_csv('/workspace/train_annotations.csv')
    train_annotated.data = train_annotated.data.apply(eval)

    train_annotated = train_annotated.groupby(['StudyInstanceUID']).agg({
        'label': list,
        'data': list
    })

    train = train.merge(train_annotated, how='left', left_on='StudyInstanceUID', right_index=True)
    train['file_path'] = train.StudyInstanceUID.apply(lambda x: f'/workspace/train/{x}.jpg')
    train['is_annotated'] = (~train['label'].isnull()).astype(int)
    return train


def get_train_folds(train):
    targets = np.dot(train[target_cols + ['is_annotated']], [2**i for i in range(12)])
    folds = list(StratifiedKFold(n_splits=5, random_state=config.seed, shuffle=True).split(X=targets, y=targets))
    return folds


def filter_train_annotated_folds(train, folds):
    ignored_index = train[train.is_annotated == 0].index.values
    folds_annotated = [(np.setdiff1d(fold[0], ignored_index), np.setdiff1d(fold[1], ignored_index)) for fold in folds]
    return folds_annotated



transforms_soft = [
    A.RandomResizedCrop(config.image_size, config.image_size, scale=(0.85, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2, brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2)),
    A.ShiftScaleRotate(p=0.2, shift_limit=0.0625, scale_limit=0.2, rotate_limit=20),
    A.CoarseDropout(p=0.2),
    A.Cutout(p=0.2, max_h_size=16, max_w_size=16, num_holes=16, fill_value=(0.)),
    A.Normalize(
        mean=[0.485],
        std=[0.229],
    ),
]


transforms_hard = [
    A.RandomResizedCrop(config.image_size, config.image_size, scale=(0.85, 1), p=1), 
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2, brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2)),
    A.ShiftScaleRotate(p=0.2, shift_limit=0.0625, scale_limit=0.2, rotate_limit=20),
    A.CLAHE(clip_limit=(1, 4), p=0.5),
    A.OneOf([
       A.OpticalDistortion(distort_limit=1.0),
       A.GridDistortion(num_steps=5, distort_limit=1.),
       A.ElasticTransform(alpha=3),
    ], p=0.2),
    A.OneOf([
       A.GaussNoise(var_limit=[10, 50]),
       A.GaussianBlur(),
       A.MotionBlur(),
       A.MedianBlur(),
    ], p=0.2),
    A.OneOf([
      A.JpegCompression(),
      A.Downscale(scale_min=0.1, scale_max=0.15),
    ], p=0.2),
    A.IAAPiecewiseAffine(p=0.2),
    A.IAASharpen(p=0.2),
    A.CoarseDropout(p=0.2),
    A.Cutout(p=0.2, max_h_size=16, max_w_size=16, num_holes=16, fill_value=(0.)),
    A.Normalize(
        mean=[0.485],
        std=[0.229],
    ),
]


class RANZCRDataset(Dataset):
    COLOR_MAP = {
        'ETT - Abnormal': (255, 0, 0),
        'ETT - Borderline': (0, 255, 0),
        'ETT - Normal': (0, 0, 255),
        'NGT - Abnormal': (255, 255, 0),
        'NGT - Borderline': (255, 0, 255),
        'NGT - Incompletely Imaged': (0, 255, 255),
        'NGT - Normal': (128, 0, 0),
        'CVC - Abnormal': (0, 128, 0),
        'CVC - Borderline': (0, 0, 128),
        'CVC - Normal': (128, 128, 0),
        'Swan Ganz Catheter Present': (128, 0, 128),
    }
    COLOR_MAP = {k: cv2.cvtColor(np.uint8(v)[None, None], cv2.COLOR_BGR2GRAY)[0] for k, v in COLOR_MAP.items()}

    def __init__(self, df, ret_mode, transform=None):
        self.df = df
        self.ret_mode = ret_mode
        self.transform = transform
        self.labels = df[target_cols].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.loc[index]
        labels = self.labels[index]
        img = cv2.imread(row.file_path, cv2.IMREAD_GRAYSCALE)
        img = np.expand_dims(img, -1)
        orig_img = img.copy() if self.ret_mode != 'orig' else img

        if self.ret_mode != 'orig' and row.is_annotated:
            for color_label, coord in zip(row.label, row.data):
                for d in coord:
                    img[d[1]-config.annot_size//2:d[1]+config.annot_size//2,
                          d[0]-config.annot_size//2:d[0]+config.annot_size//2,
                          :] = self.COLOR_MAP[color_label]

        if self.ret_mode == 'both':
            res = self.transform(image=img, orig=orig_img)
            orig_img = res['orig']
            orig_img = orig_img.transpose(2, 0, 1)
        elif self.ret_mode == 'orig':
            res = self.transform(image=orig_img)
        elif self.ret_mode == 'annotated':
            res = self.transform(image=img)

        img = res['image']
        img = img.transpose(2, 0, 1)

        if self.ret_mode == 'both':
            return (orig_img, img), labels
        else:
            return img, labels
