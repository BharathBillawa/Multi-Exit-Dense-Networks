import os
import cv2
import numpy as np
import torchvision
from torch.utils import data

TRAIN_IMG_FOLDER = 'train_rgb/'
TEST_IMG_FOLDER = 'test_rgb/'

TRAIN_SEG_FOLDER = 'train_seg13/'
TEST_SEG_FOLDER = 'test_seg13/'

TRAIN_DEPTH_FOLDER = 'train_depth/'
TEST_DEPTH_FOLDER = 'test_depth/'

TARGET_TYPE_SEMANTIC = 'semantic'
TARGET_TYPE_DEPTH = 'depth'
TARGET_TYPE_ALL = {TARGET_TYPE_SEMANTIC, TARGET_TYPE_DEPTH}

class NYUDataset(data.Dataset):
    """PyTorch wrapper class for NYU Dataset.
    Highlights: 
          ¤ Segmentation labels are mapped to 13 target classes.
          ¤ Available target types: 'semantic' and 'depth'
          ¤ 'split' specifies data split, which is either 'train' or 'val'

    Required folder structure:
          root_path:
             |-------> TRAIN_IMG_FOLDER
             |-------> TEST_IMG_FOLDER
             |
             |-------> TRAIN_SEG_FOLDER
             |-------> TEST_SEG_FOLDER
             |
             |-------> TRAIN_DEPTH_FOLDER
             |-------> TEST_DEPTH_FOLDER

    """
    def __init__(self, root_path, target_types = TARGET_TYPE_ALL, split = 'train'):

        self.root_path = root_path
        self.split = split
        self.to_tensor = torchvision.transforms.ToTensor()
        self.target_types = set(target_types)
        assert not any(self.target_types - TARGET_TYPE_ALL), 'Invalid target types!'
        
        if split == 'train':
            self.files = os.listdir(os.path.join(root_path, TRAIN_IMG_FOLDER))
            self.img_folder = TRAIN_IMG_FOLDER
            self.seg_folder = TRAIN_SEG_FOLDER
            self.depth_folder = TRAIN_DEPTH_FOLDER
        else:
            self.files = os.listdir(os.path.join(root_path, TEST_IMG_FOLDER))
            self.img_folder = TEST_IMG_FOLDER
            self.seg_folder = TEST_SEG_FOLDER
            self.depth_folder = TEST_DEPTH_FOLDER

    def __getitem__(self, index: int):

        file = self.files[index]

        img = cv2.imread(
            os.path.join(self.root_path, self.img_folder, file), 
            cv2.IMREAD_COLOR
        )
        img = self.to_tensor(img)
        
        targets = list()
        for target_type in self.target_types:
            if target_type == TARGET_TYPE_SEMANTIC:
                target_folder = self.seg_folder
            elif target_type == TARGET_TYPE_DEPTH:
                target_folder = self.depth_folder

            target = cv2.imread(
                os.path.join(self.root_path, target_folder, file), 
                cv2.IMREAD_GRAYSCALE
            )
            target = np.array(target).astype('int32')
            target = self.to_tensor(target)
            targets.append(target)

        return img, targets


    def __len__(self):
        return len(self.files)
