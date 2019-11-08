import torch
import torchvision
import os
import io
import h5py
import numpy as np
import pandas as pd
from PIL import Image
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset

class CUBDataset(Dataset):
    def __init__(self, dataset_dir, transform=None, split=0):
        self.transform = transform
        self.split = split
        self.dataset_dir = dataset_dir
        self.dataset = None
        self.dataset_keys = None
        if(split==0):
            split='train'
        elif(split==1):
            split='valid'
        else:
            split='test'

    def __len__(self):
        file = h5py.File(self.dataset_dir, 'r')
        self.dataset_keys = [str(k) for k in list(file[self.split].keys())]
        length = len(file[self.split])
        file.close()

        return length

    def __getitem__(self, idx):
        if(self.dataset==None):
            self.dataset = h5py.File(self.dataset_dir, 'r')
            self.dataset_keys = [str(k) for k in list(self.dataset[self.split].keys())]

        sample_name = self.dataset_keys[idx]
        sample = self.dataset[self.split][sample_name]

        correct_image = bytes(np.array(sample['img']))
        correct_embed = np.array(sample['embeddings'], dtype='float')
        incorrect_image = bytes(np.array(self.getIncorrectImage(sample['class'])))
        inter_embed = np.array(self.getInterEmbed())

        correct_image = Image.open(io.BytesIO(correct_image)).resize((64, 64))
        incorrect_image = Image.open(io.BytesIO(incorrect_image)).resize((64, 64))

        caption = np.array(sample['txt']).astype(str)

        sample = {'correct_image' : correct_image, 'correct_embed':correct_embed, 'incorrect_image':incorrect_image, 'inter_embed':inter_embed, 'caption':caption}
        sample['correct_image'] = sample['correct_images'].sub_(127.5).div_(127.5) #normalize
        sample['incorrect_image'] =sample['incorrect_images'].sub_(127.5).div_(127.5) #normalize

        return sample

    def getIncorrectImage(self, class_name):
        idx = np.random.randint(0, len(self.dataset_keys))
        sample_name = self.dataset_keys[idx]
        sample = self.dataset[self.split][sample_name]
        if(sample['class']!=class_name):
            return sample
        return self.getIncorrectImage(class_name)

    def getInterEmbed(self):
        idx = np.random.randint(len(self.dataset_keys))
        sample_name = self.dataset_keys[idx]
        sample = self.dataset[self.split][sample_name]
        return sample['embeddings']

    def qualityInsurance(self, img):
        img = np.array(img, dtype=float)
        if len(img.shape) < 3:
            rgb = np.empty((64, 64, 3), dtype=np.float32)
            rgb[:, :, 0] = img
            rgb[:, :, 1] = img
            rgb[:, :, 2] = img
            img = rgb

        return img.transpose(2, 0, 1)
