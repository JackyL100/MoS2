import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import time
from torchvision.transforms.functional import to_pil_image
import os

color_map = {
    (0,0,0) : 0,
    (60, 28, 255) : 1,
    (73, 255, 52) : 2,
    (255, 70, 70) : 3,
    (7, 255, 251) : 4,
    (88, 88, 88) : 5
}

def get_coverage(mask):
    if isinstance(mask, Image.Image):
        return mask.getcolors()
    if isinstance(mask, np.ndarray):
        assert mask.shape[2] == 3
        return Image.fromarray(mask, 'RGB').getcolors()
    if isinstance(mask, torch.Tensor):
        assert mask.shape[0] == 3
        return to_pil_image(mask).getcolors()

if __name__ == "__main__":
    data = []
    for file in os.scandir('../Image Segmentation Data/masks/'):
        mask = Image.open(file.path)
        coverage = get_coverage(mask)
        mask_info = {}
        mask_info['name'] = file.name
        mask_info['Layer Coverage'] = {}
        for pixel_count, color in coverage:
            layer_type = color_map[color]
            if (layer_type == 0):
                continue
            mask_info['Layer Coverage'][layer_type] = pixel_count / (mask.width * mask.height)
        data.append(mask_info)
    print(json.dumps(data, indent=4))
    with open("../Data/coverage.json", "w") as f:
        f.write(json.dumps(data, indent=4))