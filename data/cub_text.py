import os, re
from math import radians, cos, sin, pi
import numpy as np
from PIL import Image
import pandas as pd
import torch.utils.data as data
from model.clip.clip import tokenize
    
    
def get_dataframe(root, train=True):
    """ Get meta infomation from txt and return DataFrame

    Args:
        root (str): 'root dir of CUB dataset'
        train (bool): whether is training set.

    """
    root = os.path.join(root, 'CUB_200_2011')
    # read from file
    df_img_name = pd.read_csv(os.path.join(root, 'images.txt'), sep=' ', header=None, names=['image_id', 'image_name'])
    df_class_label = pd.read_csv(os.path.join(root, 'image_class_labels.txt'), sep=' ', header=None, names=['image_id', 'class_id'])
    df_train = pd.read_csv(os.path.join(root, 'train_test_split.txt'), sep=' ', header=None, names=['image_id', 'is_training_image'])
    # split train/text
    df_img_name = train_text_split(df_img_name, df_train, train)
    df_class_label = train_text_split(df_class_label, df_train, train)

    return df_img_name, df_class_label


def train_text_split(df, df_train, is_train=True):
    df = pd.merge(df, df_train, on='image_id')
    df = df.loc[df['is_training_image'] == int(is_train)]
    del df['is_training_image']
    return df

def index_from_zero(df, index_name='image_id'):
    df[index_name] = df[index_name].apply(lambda x: x - 1)
    return df





class CUBDataset(data.Dataset):
    def __init__(self, root, transform=None, text_transform=None, train=True):
        
        self.root = root
        img_root = os.path.join(root, 'CUB_200_2011', 'images')
        text_root = os.path.join(root, 'cvpr2016_cub', 'text_c10')
        self.transform = transform
        self.text_transform = text_transform
        self.train = train

        df_img_name, df_class_label = get_dataframe(root, train)
        self.img_paths = [os.path.join(img_root, name) for name in df_img_name['image_name']]
        self.text_paths = [os.path.join(text_root, name.replace('.jpg', '.txt')) for name in df_img_name['image_name']]
        self.targets = list(df_class_label['class_id'])

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        target = self.targets[idx]
        with open(self.text_paths[idx], mode="r") as f:
            idx = np.random.randint(0, 10)
            text = f.readlines()[idx]

        if self.transform is not None:
            img = self.transform(img)

        if self.text_transform:
            text = self.text_transform(text)
        
        text = tokenize(text).squeeze()

        return img, text, target

    def __len__(self):
        return len(self.targets)
