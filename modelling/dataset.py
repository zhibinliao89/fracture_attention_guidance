"""
adapted from: https://github.com/zhangyongshun/resnet_finetune_cub/blob/master/cub.py
"""
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import pickle
import tarfile
from collections import defaultdict
import pandas as pd
import cv2
from tqdm import tqdm
from torchvision import transforms


def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])


def selector(label, num_images_per_class):
    selections = list()
    for group_id, group_df in pd.DataFrame(label, columns=['label']).groupby('label'):
        order = np.random.permutation(len(group_df))
        selections.append(list(group_df.index[order[:num_images_per_class]]))
    selections = np.concatenate(selections).tolist()
    return selections


class CUB200(Dataset):
    def __init__(self, root="processed", train=True, transform=None, config=None):
        super(CUB200, self).__init__()

        self.root = root
        self.train = train
        self.transform = transform
        self.normalization = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.config = config

        if self._check_processed():
            print('Train file has been extracted' if self.train else 'Test file has been extracted')
        else:
            self._extract()

        if self._check_parts_processed():
            print('Train parts file has been extracted' if self.train else 'Test parts file has been extracted')
        else:
            self.train_data, self.train_label = pickle.load(
                open(os.path.join(self.root, 'processed/train.pkl'), 'rb')
            )
            self.test_data, self.test_label = pickle.load(
                open(os.path.join(self.root, 'processed/test.pkl'), 'rb')
            )
            self._extract_parts()
            del self.train_data, self.train_label, self.test_data, self.test_label

        if self.train:
            self.train_data, self.train_label = pickle.load(
                open(os.path.join(self.root, 'processed/train.pkl'), 'rb')
            )
            self.train_parts = pickle.load(
                open(os.path.join(self.root, 'processed/train_parts.pkl'), 'rb')
            )
            selections = selector(self.train_label, self.config['num_training_images_per_class'])
            self.train_data = [self.train_data[s] for s in selections]
            self.train_label = [self.train_label[s] for s in selections]
            self.train_parts = {idx_s: self.train_parts[s] for idx_s, s in enumerate(selections)}
        else:
            self.test_data, self.test_label = pickle.load(
                open(os.path.join(self.root, 'processed/test.pkl'), 'rb')
            )
            self.test_parts = pickle.load(
                open(os.path.join(self.root, 'processed/test_parts.pkl'), 'rb')
            )

    def __len__(self):
        return len(self.train_data) if self.train else len(self.test_data)

    def __getitem__(self, idx):
        if self.train:
            img, label, mask = self.train_data[idx], self.train_label[idx], self.train_parts[idx]
        else:
            img, label, mask = self.test_data[idx], self.test_label[idx], self.test_parts[idx]
        img = Image.fromarray(img)
        mask = Image.fromarray(mask)
        if self.transform is not None:
            img, mask = self.transform(img, mask)
        source_img = img
        img = self.normalization(img)
        sel = mask == 0
        mask[~sel] = 1.
        mask[sel] = self.config['ignore_value']
        label = one_hot(np.array([label]), 200)
        return source_img, img, label, mask

    def _check_processed(self):
        assert os.path.isdir(self.root)
        assert os.path.isfile(os.path.join(self.root, 'CUB_200_2011.tgz'))
        return (os.path.isfile(os.path.join(self.root, 'processed/train.pkl')) and
                os.path.isfile(os.path.join(self.root, 'processed/test.pkl')))

    def _check_parts_processed(self):
        return (os.path.isfile(os.path.join(self.root, 'processed/train_parts.pkl')) and
                os.path.isfile(os.path.join(self.root, 'processed/test_parts.pkl')))

    def _extract(self):
        processed_data_path = os.path.join(self.root, 'processed')
        if not os.path.isdir(processed_data_path):
            os.mkdir(processed_data_path)

        cub_tgz_path = os.path.join(self.root, 'CUB_200_2011.tgz')
        images_txt_path = 'CUB_200_2011/images.txt'
        train_test_split_txt_path = 'CUB_200_2011/train_test_split.txt'

        tar = tarfile.open(cub_tgz_path, 'r:gz')
        images_txt = tar.extractfile(tar.getmember(images_txt_path))
        train_test_split_txt = tar.extractfile(tar.getmember(train_test_split_txt_path))
        if not (images_txt and train_test_split_txt):
            print('Extract image.txt and train_test_split.txt Error!')
            raise RuntimeError('cub-200-1011')

        images_txt = images_txt.read().decode('utf-8').splitlines()
        train_test_split_txt = train_test_split_txt.read().decode('utf-8').splitlines()

        id2name = np.genfromtxt(images_txt, dtype=str)
        id2train = np.genfromtxt(train_test_split_txt, dtype=int)
        print('Finish loading images.txt and train_test_split.txt')
        train_data = []
        train_labels = []
        test_data = []
        test_labels = []
        print('Start extract images..')
        cnt = 0
        train_cnt = 0
        test_cnt = 0
        for _id in range(id2name.shape[0]):
            cnt += 1

            image_path = 'CUB_200_2011/images/' + id2name[_id, 1]
            image = tar.extractfile(tar.getmember(image_path))
            if not image:
                print('get image: ' + image_path + ' error')
                raise RuntimeError
            image = Image.open(image)
            label = int(id2name[_id, 1][:3]) - 1

            if image.getbands()[0] == 'L':
                image = image.convert('RGB')
            image_np = np.array(image)
            image.close()

            if id2train[_id, 1] == 1:
                train_cnt += 1
                train_data.append(image_np)
                train_labels.append(label)
            else:
                test_cnt += 1
                test_data.append(image_np)
                test_labels.append(label)
            if cnt % 500 == 0:
                print('{} images have been extracted'.format(cnt))
        print('Total images: {}, training images: {}. testing images: {}'.format(cnt, train_cnt, test_cnt))
        tar.close()
        pickle.dump((train_data, train_labels),
                    open(os.path.join(self.root, 'processed/train.pkl'), 'wb'))
        pickle.dump((test_data, test_labels),
                    open(os.path.join(self.root, 'processed/test.pkl'), 'wb'))

    def _extract_parts(self):
        processed_data_path = os.path.join(self.root, 'processed')
        if not os.path.isdir(processed_data_path):
            os.mkdir(processed_data_path)

        cub_tgz_path = os.path.join(self.root, 'CUB_200_2011.tgz')
        images_txt_path = 'CUB_200_2011/images.txt'
        train_test_split_txt_path = 'CUB_200_2011/train_test_split.txt'
        part_locs_txt_path = 'CUB_200_2011/parts/part_locs.txt'

        tar = tarfile.open(cub_tgz_path, 'r:gz')
        images_txt = tar.extractfile(tar.getmember(images_txt_path))
        train_test_split_txt = tar.extractfile(tar.getmember(train_test_split_txt_path))
        part_locs_txt = tar.extractfile(tar.getmember(part_locs_txt_path))

        images_txt = images_txt.read().decode('utf-8').splitlines()
        train_test_split_txt = train_test_split_txt.read().decode('utf-8').splitlines()
        part_locs_txt = part_locs_txt.read().decode('utf-8').splitlines()

        id2name = np.genfromtxt(images_txt, dtype=str)
        id2name = pd.DataFrame(id2name, columns=['original_index', 'image_name'])
        id2name['original_index'] = id2name['original_index'].astype(int)
        id2name = id2name.set_index('original_index')
        id2train = np.genfromtxt(train_test_split_txt, dtype=int)
        id2train = pd.DataFrame(id2train, columns=['original_index', 'set'])
        id2train = id2train.set_index('original_index')

        info = id2name.join(id2train, on='original_index')
        train_info = info.loc[info['set'] == 1]
        train_info = train_info.reset_index()
        train_info['new_index'] = train_info.index
        train_info = train_info.set_index('original_index')
        test_info = info.loc[info['set'] == 0]
        test_info = test_info.reset_index()
        test_info['new_index'] = test_info.index
        test_info = test_info.set_index('original_index')

        id2part = np.genfromtxt(part_locs_txt, dtype=float)
        print('Finish loading images.txt and train_test_split.txt')

        print('Start extract parts location data..')
        train_parts = defaultdict(None)
        test_parts = defaultdict(None)
        for _id, (image_id, class_id, x, y, visible) in tqdm(enumerate(id2part)):
            image_id = int(image_id)
            # class_id = int(class_id)
            visible = int(visible)
            if visible:
                if id2train.loc[image_id]['set'] == 1:
                    train_index = train_info.loc[image_id]['new_index']
                    if train_index not in train_parts:
                        mask = np.zeros(self.train_data[train_index].shape[:2], dtype=np.uint8)
                        train_parts[train_index] = mask
                    else:
                        mask = train_parts[train_index]
                else:
                    test_index = test_info.loc[image_id]['new_index']
                    if test_index not in test_parts:
                        mask = np.zeros(self.test_data[test_index].shape[:2], dtype=np.uint8)
                        test_parts[test_index] = mask
                    else:
                        mask = test_parts[test_index]
                cv2.circle(mask, (int(x), int(y)), radius=50, color=1, thickness=-1)

        tar.close()
        pickle.dump(train_parts,
                    open(os.path.join(self.root, 'processed/train_parts.pkl'), 'wb'))
        pickle.dump(test_parts,
                    open(os.path.join(self.root, 'processed/test_parts.pkl'), 'wb'))

