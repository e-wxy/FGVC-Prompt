import os, re
from math import radians, cos, sin, pi
import numpy as np
from PIL import Image
import pandas as pd
import torch.utils.data as data
    
    
def get_dataframe(root, train=True):
    attri_dir = os.path.join(root, 'CUB_200_2011', 'attributes')
    attri_file = os.path.join(attri_dir, 'attributes.txt')
    label_file = os.path.join(attri_dir, 'image_attribute_labels_clean.txt') # `image_attribute_labels_clean.txt` remove the rudundent '0' from `image_attribute_labels.txt`
    # read from file
    df_img_name = pd.read_csv(os.join(root, 'images.txt'), sep=' ', header=None, names=['image_id', 'image_name'])
    df_class_label = pd.read_csv(os.join(root, 'image_class_labels.txt'), sep=' ', header=None, names=['image_id', 'class_id'])
    df_attri = pd.read_csv(attri_file, sep=' ', header=None, names=['attribute_id', 'description'])
    df_attri_label = pd.read_csv(label_file, sep=' ', header=None, names=['image_id', 'attribute_id', 'is_present', 'certainty_id', 'time'])
    df_train = pd.read_csv(os.join(root, 'train_text_split.txt'), sep=' ', header=None, names=['image_id', 'is_training_image'])
    # pre-process
    # attribute -> text
    df_attri['description'] = df_attri['description'].apply(lambda x: x.replace('has_', '').replace('_', ' ').replace('::', ' is '))
    # select presented attributes
    df_attri_label = df_attri_label.loc[df_attri_label['is_present'] == 1]
    # split train/text
    df_img_name = train_text_split(df_img_name, df_train, train)
    df_class_label = train_text_split(df_class_label, df_train, train)
    df_attri_label = train_text_split(df_attri_label, df_train, train)

    return df_img_name, df_class_label, df_attri, df_attri_label


def train_text_split(df, df_train, is_train=True):
    df = pd.merge(df, df_train, on='image_id')
    df = df.loc[df['is_training_image'] == int(is_train)]
    del df['is_training_image']
    return df

def index_from_zero(df, index_name='image_id'):
    df[index_name] = df[index_name].apply(lambda x: x - 1)
    return df


class RandomPermutateDrop(object):

    def __init__(self, drop_rate: float = 0.0):
        self.drop_rate = drop_rate

    def __call__(self, des_idxes):
        np.random.shuffle(des_idxes)
        length = len(des_idxes)
        des_idxes = des_idxes[:length - int(length * np.random.randint(self.drop_rate))]
        return des_idxes
    


def get_spatial_info(latitude, longitude):
    if latitude and longitude:
        latitude = radians(latitude)
        longitude = radians(longitude)
        x = cos(latitude)*cos(longitude)
        y = cos(latitude)*sin(longitude)
        z = sin(latitude)
        return [x, y, z]
    else:
        return [0, 0, 0]


def get_temporal_info(date, miss_hour=False):
    try:
        if date:
            if miss_hour:
                pattern = re.compile(r'(\d*)-(\d*)-(\d*)', re.I)
            else:
                pattern = re.compile(r'(\d*)-(\d*)-(\d*) (\d*):(\d*):(\d*)', re.I)
            m = pattern.match(date.strip())

            if m:
                year = int(m.group(1))
                month = int(m.group(2))
                day = int(m.group(3))
                x_month = sin(2*pi*month/12)
                y_month = cos(2*pi*month/12)
                if miss_hour:
                    x_hour = 0
                    y_hour = 0
                else:
                    hour = int(m.group(4))
                    x_hour = sin(2*pi*hour/24)
                    y_hour = cos(2*pi*hour/24)
                return [x_month, y_month, x_hour, y_hour]
            else:
                return [0, 0, 0, 0]
        else:
            return [0, 0, 0, 0]
    except:
        return [0, 0, 0, 0]



class CUBDataset(data.Dataset):
    def __init__(self, root, transform=None, text_transform=RandomPermutateDrop(), train=True, 
                 prompt_start="A photo of a bird, of which the ", 
                 prompt_link=", the "):
        
        self.root = root
        img_root = os.path.join(root, 'CUB_200_2011', 'images')
        self.transform = transform
        self.text_transform = text_transform
        self.train = train
        self.prompt_start = prompt_start
        self.prompt_link = prompt_link
        df_img_name, df_class_label, df_attri, df_attri_label = get_dataframe(root, train)
        # self.img_paths = list(df_img_name['image_name'])
        self.img_paths = [os.path.join(img_root, name) for name in df_img_name['image_name']]
        self.targets = list(df_class_label['class_id'])
        self.df_attri = df_attri
        self.df_attri_label = index_from_zero(df_attri_label)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        target = self.targets[idx]
        des_list = list(self.df_attri_label.loc[self.df_attri_label['image_id'] == idx])
        des_idxes = np.arange(len(des_list))

        if self.transform is not None:
            img = self.transform(img)

        if self.text_transform:
            des_idxes = self.text_transform(des_idxes)

        text = self.prompt_start + self.prompt_link.join([des_list[des_idx] for des_idx in des_idxes]) + "."

        return img, text, target

    def __len__(self):
        return len(self.targets)
