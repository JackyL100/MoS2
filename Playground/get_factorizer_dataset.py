import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.utils
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.io import decode_image
from PIL import Image
import os
import matplotlib.pyplot as plt

device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

class ImageDataset(Dataset):
    def __init__(self, image_folder, mask_folder):
        super(ImageDataset).__init__()
        self.length = 0
        self.image_path = image_folder
        self.mask_path = mask_folder
        self.color_mapping = {
            (0,0,0) : 0,
            (60, 28, 255) : 1,
            (73, 255, 52) : 2,
            (255, 70, 70) : 3,
            (7, 255, 251) : 4,
            (88, 88, 88) : 5
        }
        for path in os.listdir(f'../Image Segmentation Data/{self.image_path}'):
            if os.path.isfile(os.path.join(f'../Image Segmentation Data/{self.image_path}', path)):
                self.length += 1
    def process_mask(self, mask):
        masks = torch.zeros((6,588,780)).to(device)

        m = mask.permute(1,2,0)

        for color, idx in self.color_mapping.items():
            c = torch.tensor(color,dtype=m.dtype, device=device)
            matches = (m == c).all(dim=-1)
            masks[idx][matches] = 1
        return masks

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        i = 0
        for image in os.scandir(f'../Image Segmentation Data/{self.image_path}'):
            if index == i:
                img = decode_image(image.path).to(torch.float32).to(device)
                mask = decode_image(f'../Image Segmentation Data/{self.mask_path}/{image.name[:-3] + 'png'}').to(torch.float32).to(device)
                mask = self.process_mask(mask)
            i+=1
        return img[:, 6:-6, 6:-6], mask[:, 6:-6, 6:-6]
    
def get_dataloader(folder_path, batch_size):
    split = [0.8, 0.2]
    dataset = ImageDataset(folder_path, 'masks')
    train_ds, test_ds = torch.utils.data.random_split(dataset, split)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader

if __name__ == "__main__":
    train, test = get_dataloader('normalized_red_quantized', 1)
    img, mask = next(iter(train))
    print(img.shape)
    print(mask.shape)
    plt.imshow(mask[0][1].cpu())
    plt.show()