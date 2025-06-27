import torch 
import pandas as pd
import numpy as np
import torch.utils
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.io import decode_image
from PIL import Image
import os

class SegmentationDataset(Dataset):
    def __init__(self, images_path, masks_path):
        super().__init__()
        self.transform = None
        self.images_path = images_path
        self.masks_path = masks_path
    def __len__(self):
        return len([file for file in os.scandir(f'../Image Segmentation Data/{self.images_path}')])
    def __getitem__(self, index):
        img = Image.new('RGB',(588,780))
        mask = Image.new('RGB',(588,780))
        i = 0
        for image in os.scandir(f'../Image Segmentation Data/{self.images_path}'):
            if index == i:
                img = Image.open(image.path)
                mask = Image.open(f'../Image Segmentation Data/{self.masks_path}/{image.name[:-3] + 'png'}')
                break
            i += 1
        if self.transform != None:
            img = self.transform(img)
            mask = self.transform(mask)
        return np.array(img), np.array(mask)
        
def get_dataloader(name: str):
    dataset = SegmentationDataset(name, 'masks')
    train_ds, test_ds = torch.utils.data.random_split(dataset, [0.8,0.2])
    train_loader = DataLoader(train_ds, batch_size=1)
    test_loader = DataLoader(test_ds, batch_size=1)
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
    # width, height = mask.size
    # colors = [0 for _ in range(6)]
    # for i in range(width):
    #     for j in range(height):
    #         colors[color_mapping[cropped_mask.getpixel((i,j))]] += 1
    # return np.argmax(colors)
    assert len(cropped_mask.shape) == 4
    colors = [[0 for _ in range(6)] for _ in range(0,cropped_mask.shape[0])]
    for mask in range(cropped_mask.shape[0]):
        for i in range(cropped_mask.shape[2]):
            for j in range(cropped_mask.shape[3]):
                color = tuple(cropped_mask[mask, :, i, j].tolist())
                colors[mask][color_mapping[color]] += 1
    return np.argmax(colors)

if __name__ == "__main__":
    file_path = "Image Segmentation Data/masks/4_MoS2_4.png"
    mask = Image.open(file_path)
    cropped_mask = T.functional.crop(T.functional.pil_to_tensor(mask), 0,0,12,12)
    print(get_label(cropped_mask))