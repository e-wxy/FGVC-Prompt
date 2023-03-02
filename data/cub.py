import os, re, pickle
from math import radians, cos, sin, pi
import numpy as np
from PIL import Image
import torch.utils.data as data

IMG_EXTENSIONS = ['.png', '.jpg', '.jpeg']

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


def find_images_and_targets(root, istrain=False, aux_info=False):
    imageid2label = {}
    with open(os.path.join(os.path.join(root, 'CUB_200_2011'), 'image_class_labels.txt'), 'r') as f:
        for line in f:
            image_id, label = line.split()
            imageid2label[int(image_id)] = int(label)-1
    imageid2split = {}
    with open(os.path.join(os.path.join(root, 'CUB_200_2011'), 'train_test_split.txt'), 'r') as f:
        for line in f:
            image_id, split = line.split()
            imageid2split[int(image_id)] = int(split)

    images_root = os.path.join(os.path.join(root, 'CUB_200_2011'), 'images')
    images_paths = []
    targets = []
    meta = []

    # Attribution
    if aux_info == 'attri':
        attributes_root = os.path.join(os.path.join(root, 'CUB_200_2011'), 'attributes')
        imageid2attribute = {}
        with open(os.path.join(attributes_root, 'image_attribute_labels.txt'), 'r') as f:
            for line in f:
                if len(line.split()) == 6:
                    image_id, attribute_id, is_present, _, _, _ = line.split()
                else:
                    image_id, attribute_id, is_present, certainty_id, time = line.split()
                if int(image_id) not in imageid2attribute:
                    imageid2attribute[int(image_id)] = [0 for i in range(312)]
                imageid2attribute[int(image_id)][int(attribute_id)-1] = int(is_present)
    # Text Description
    if aux_info == 'text':
        bert_embedding_root = os.path.join(root, 'bert_embedding_cub')
        # text_root = os.path.join(root, 'text_c10')
        with open(os.path.join(bert_embedding_root, file_name.replace('.jpg', '.pickle')), 'rb') as f_bert:
            bert_embedding = pickle.load(f_bert)
            bert_embedding = bert_embedding['embedding_words']
        
    with open(os.path.join(os.path.join(root, 'CUB_200_2011'), 'images.txt'), 'r') as f:
        for line in f:
            image_id, file_name = line.split()
            file_path = os.path.join(images_root, file_name)
            target = imageid2label[int(image_id)]

            if (istrain and imageid2split[int(image_id)] == 1) or (not istrain and imageid2split[int(image_id)] == 0):
                images_paths.append(file_path)
                targets.append(target)
                if aux_info:
                    if aux_info == 'text':
                        meta.append(bert_embedding)
                    elif aux_info == 'attri':
                        meta.append(imageid2attribute[int(image_id)])
                    
    return images_paths, targets, meta

class CUBDataset(data.Dataset):
    def __init__(self, root, transform, meta_type=None, train=True):
        self.root = root
        self.transform = transform
        self.meta_type = meta_type
        self.train = train
        self.img_paths, self.targets, self.meta = find_images_and_targets(root, train, meta_type)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        target = self.targets[idx]

        if self.transform is not None:
            img = self.transform(img)
        if self.meta_type is not None:
            meta_info = self.meta[idx]
            if type(meta_info) is np.ndarray:   # random select a text description
                select_index = np.random.randint(meta_info.shape[0])
                return img, target, meta_info[select_index, :]
            else:
                return img, target, np.asarray(meta_info).astype(np.float64)
        else:
            return img, target

    def __len__(self):
        return len(self.targets)


if __name__ == '__main__':
    find_images_and_targets('./cub2002011')
