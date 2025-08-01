import torch 
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import torch.utils
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms as T
from torchvision.io import decode_image
from PIL import Image
import os
import matplotlib.pyplot as plt

class SegmentationDataset(Dataset):
    def __init__(self, images_path, masks_path, patch_size):
        super().__init__()
        self.transform = None
        self.images_path = images_path
        self.masks_path = masks_path

        self.img_height = 588
        self.img_width = 780
        assert self.img_height % patch_size == 0
        assert self.img_width % patch_size == 0
        self.patch_size = patch_size
        self.num_images = len([file for file in os.scandir(f'../Image Segmentation Data/{self.images_path}')])
        self.num_img_patches = int((self.img_height / self.patch_size) * (self.img_width / self.patch_size))
    def __len__(self):
        return int(self.num_images * (self.img_height / self.patch_size) * (self.img_width / self.patch_size))
    def __getitem__(self, index):
        img_num = index // self.num_img_patches
        patch_num = index % self.num_img_patches
        row = patch_num // self.img_width
        col = patch_num % self.img_height
        img_patch = torch.empty((3,12,12))
        label_dist = torch.empty((6))
        i = 0
        for image in os.scandir(f'../Image Segmentation Data/{self.images_path}'):
            if img_num == i:
                pil_img = Image.open(image.path)
                pil_mask = Image.open(f'../Image Segmentation Data/{self.masks_path}/{image.name[:-3] + 'png'}')
                img = T.functional.pil_to_tensor(pil_img)
                mask = T.functional.pil_to_tensor(pil_mask)
                img = torch.transpose(img, 1,2)
                mask = torch.transpose(mask, 1, 2)
                box = (row, col, self.patch_size, self.patch_size)
                # crop out patch from image and mask
                img_patch = T.functional.crop(img, box[0], box[1], box[2], box[3]).to(torch.float32)
                mask_patch = T.functional.crop(mask, box[0], box[1], box[2], box[3])
                # get the label of the patch based on dominant class in patch
                label = get_label(mask_patch)
                label_tensor = torch.tensor(label).to(torch.int64)
                label_dist = nn.functional.softmax(nn.functional.one_hot(label_tensor, num_classes=6).to(torch.float32))
                # label_dist = nn.functional.one_hot(label_tensor, num_classes=6).to(torch.float32)
                break
            i += 1
        return img_patch, label_dist
        
def get_dataloader(name: str, patch_size: int, batch_size: int):

    split = [0.8, 0.2]
    dataset = SegmentationDataset(name, 'masks', patch_size)
    train_ds, test_ds = torch.utils.data.random_split(dataset, split)
    train_labels = []
    for i in range(len(train_ds)):
        train_labels.append(np.argmax(train_ds[i][1]))
    class_counts = torch.bincount(torch.tensor(train_labels))
    weights = 1 / class_counts[torch.tensor(train_labels)]
    weights = weights.float()
    train_sampler = WeightedRandomSampler(weights=weights, num_samples=len(train_ds))

    # print(len(dataset))
    # print(len(train_ds))
    # print(len(test_ds))
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader

color_mapping = {
    (0,0,0) : 0,
    (60, 28, 255) : 1,
    (73, 255, 52) : 2,
    (255, 70, 70) : 3,
    (7, 255, 251) : 4,
    (88, 88, 88) : 5
}

def get_label(cropped_mask):
    assert len(cropped_mask.shape) == 3
    colors = [0 for _ in range(6)]
    for i in range(cropped_mask.shape[1]):
        for j in range(cropped_mask.shape[2]):
            color = tuple(cropped_mask[:, i, j].tolist())
            colors[color_mapping[color]] += 1
    return np.argmax(colors)

def count_labels(loader, arr):
    for batch in loader:
        _, labels = batch
        labels = torch.argmax(labels, dim=1)
        for label in labels:
            l = int(label)
            arr[l] += 1
    print(arr)
    return arr

if __name__ == "__main__":
    train, test = get_dataloader('normalized_red_quantized', patch_size=12, batch_size=100)

    # train_counts = [0 for _ in range(6)]
    # test_counts = [0 for _ in range(6)]

    # train_counts = count_labels(train, train_counts)
    # test_counts = count_labels(test, test_counts)