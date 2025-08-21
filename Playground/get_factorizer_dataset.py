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
        # self.color_mapping = {
        #     (0,0,0) : 0,
        #     (60, 28, 255) : 1,
        #     (73, 255, 52) : 2,
        #     (255, 70, 70) : 3,
        #     (7, 255, 251) : 4,
        #     (88, 88, 88) : 5
        # }
        self.color_mapping = {
            (0,0,0) : 0, 
            (6,4,243) : 1, 
            (88,255,52) : 2, 
            (255,28,36) : 3, 
            (77,241,232) : 4, 
            (209,209,216) : 5
        }
        self.formats = ['png', 'jpg', 'jpeg', 'JPG']
        for path in os.listdir(f'../Image Segmentation Data/{self.image_path}'):
            if os.path.isfile(os.path.join(f'../Image Segmentation Data/{self.image_path}', path)) and path[-3:] in self.formats:
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
        return self.length * 4
    
    def __getitem__(self, index):
        i = 0
        for image in os.scandir(f'../Image Segmentation Data/{self.image_path}'):
            if image.path[-3:] not in self.formats:
                continue
            if index // 4 == i:

                try:
                    img = decode_image(image.path).to(torch.float32).to(device)
                    mask = decode_image(f'../Image Segmentation Data/{self.mask_path}/{image.name[:-3] + 'png'}').to(torch.float32).to(device)
                    masks = self.process_mask(mask)

                    if np.random.random(1)[0] > 0.5: # flip along height dimension
                        img = torch.flip(img, [1])
                        mask = torch.flip(mask, [1])
                        masks = torch.flip(masks, [1])
                    if np.random.random(1)[0] > 0.5: # flip along width dimension
                        img = torch.flip(img, [2])
                        mask = torch.flip(mask , [2])
                        masks = torch.flip(masks, [2])

                except:
                    print(image.path)
            i+=1
        return img[:, 6:-6, 6:-6], masks[:, 6:-6, 6:-6], mask[:, 6:-6, 6:-6]
    
def get_dataloader(image_folder, mask_folder, batch_size):
    split = [0.8, 0.2]
    dataset = ImageDataset(image_folder, mask_folder)
    train_ds, test_ds = torch.utils.data.random_split(dataset, split)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader

if __name__ == "__main__":
    train, test = get_dataloader('images_2', 'masks_2', 1)
    img, masks, mask = next(iter(train))
    print(img.shape)
    print(mask.shape)
    masks = masks.detach().cpu().numpy()

    fig, axs = plt.subplots(4, 2)
    fig.set_size_inches(12,12)
    fig.tight_layout()
    axs[0,0].set_title('Original')
    axs[0,0].imshow(img[0].detach().cpu().numpy().transpose(1,2,0) / 255.0)
    axs[0,1].set_title("Background Mask")
    axs[0,1].imshow(masks[0][0], cmap='gray')
    axs[1,0].set_title("Monolayer Mask")
    axs[1,0].imshow(masks[0][1], cmap='gray')
    axs[1,1].set_title("Bilayer Mask")
    axs[1,1].imshow(masks[0][2], cmap='gray')
    axs[2,0].set_title("Trilayer Mask")
    axs[2,0].imshow(masks[0][3], cmap='gray')
    axs[2,1].set_title("Four-layer Mask")
    axs[2,1].imshow(masks[0][4], cmap='gray')
    axs[3,0].set_title("Bulk Mask")
    axs[3,0].imshow(masks[0][5], cmap='gray')
    axs[3,1].set_title("Ground Truth")
    axs[3,1].imshow(mask[0].permute(1,2,0).detach().cpu().numpy()/255.0)
    plt.show()