import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import csv
from tqdm import tqdm

color_mapping = {
    '1': (60, 28, 255),
    '2': (73, 255, 52),
    '3': (255, 70, 70),
    '4': (7, 255, 251),
    '5': (88, 88, 88)
}

def get_mask_coors(filePath, color, tolerance=50):
    img_array = np.array(Image.open(filePath).convert('RGB'))
    distances = np.sqrt(np.sum((img_array - color) ** 2, axis=-1))
    
    mask = distances <= tolerance

    mask_y, mask_x = np.where(mask)
    background_y, background_x = np.where(np.invert(mask))
    return list(zip(mask_x, mask_y)), list(zip(background_x, background_y))

def create_dataset(segmentation_data_folder, src):
    with open(f"Playground/colors_{src}.csv", 'w') as f:
        writer = csv.writer(f)
        column_names = [["Red", "Green", "Blue", "X", "Y", "Layers"]]
        writer.writerows(column_names)
        for m in tqdm(os.scandir(segmentation_data_folder + "/masks")):
            for o in os.scandir(segmentation_data_folder + "/" + src):
                if m.name[:-3] != o.name[:-3]:
                    continue
                data_batch = []
                original = np.array(Image.open(o.path).convert('RGB'))
                for color in color_mapping:
                    mask_coors, background_coors = get_mask_coors(m.path, color_mapping[color])
                    for coor in mask_coors:
                        x,y = coor
                        pixel_val = original[y,x,:]
                        data_batch.append([pixel_val[0], pixel_val[1], pixel_val[2], x,y,color])
                    i = 0
                    for coor in background_coors:
                        if i % 50 == 0:
                            x,y = coor
                            pixel_val = original[y,x,:]
                            data_batch.append([pixel_val[0], pixel_val[1], pixel_val[2], x, y, 0])
                        i += 1
                writer.writerows(data_batch)

create_dataset("Image Segmentation Data", 'normalized_quantized')