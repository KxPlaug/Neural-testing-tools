import os
import pickle as pkl

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from tqdm import tqdm
import numpy as np


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class ImageNetDataset(Dataset):
    def __init__(self, train, transform):
        super().__init__()
        if train:
            raise NotImplementedError

        self.data = pd.read_csv('data/mini-imagenet/test.csv')
        self.transform = transform
        self.class_name2label = pkl.load(
            open('data/class_name2label.pkl', 'rb'))
        self.classes = list(self.class_name2label.keys())
        imgs_path = os.path.join('data', 'mini-imagenet', 'test_imgs.pt')
        labels_path = os.path.join('data', 'mini-imagenet', 'test_labels.pt')
        if not os.path.exists(imgs_path) or not os.path.exists(labels_path):
            self.imgs = []
            self.labels = []
            for i, row in tqdm(self.data.iterrows(), total=len(self.data)):
                img = pil_loader(os.path.join(
                    "data", "mini-imagenet", "images", row['filename']))
                img = self.transform(img)
                self.imgs.append(img)
                self.labels.append(int(self.class_name2label[row['label']]))
            self.imgs = torch.stack(self.imgs)
            self.labels = torch.LongTensor(self.labels)
            torch.save(self.imgs, imgs_path)
            torch.save(self.labels, labels_path)
        else:
            self.imgs = torch.load(imgs_path)
            self.labels = torch.load(labels_path)

    def __getitem__(self, index):
        return self.imgs[index], self.labels[index]

    def __len__(self):
        return len(self.data)


def get_dataset(name, batch_size):
    if name == "cifar10":
        _normalizer = transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
        dataset = datasets.CIFAR10("./data", train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            _normalizer,
        ]))
        name_map = {v:0 for v in range(10)}
    elif name == "cifar100":
        _normalizer = transforms.Normalize(
            (0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
        dataset = datasets.CIFAR100("./data", train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            _normalizer,
        ]))
        name_map = {v:0 for v in range(100)}
    elif name == "imagenet":
        _normalizer = transforms.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        dataset = ImageNetDataset(train=False, transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            _normalizer,
        ]))
        name_map = {v:0 for v in range(1000)}
    else:
        raise NotImplementedError
    data_min = np.min((0 - np.array(_normalizer.mean)) /
                      np.array(_normalizer.std))
    data_max = np.max((1 - np.array(_normalizer.mean)) /
                      np.array(_normalizer.std))
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, pin_memory=True)
    return dataloader, data_min, data_max, _normalizer, name_map