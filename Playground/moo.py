import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# def patch_image(file_path, patch_size):
#     width, height = patch_size
#     img = Image.open(file_path)
#     plt.imshow(img)
#     plt.xticks([i for i in range(0,img.size[0], width)])
#     plt.yticks([i for i in range(0,img.size[1], height)])
#     plt.grid()
#     plt.show()
# patch_image('Image Segmentation Data/denoised/4_MoS2_4.jpg',(12,12))

import cv2

segment_color_map = {
    (6,4,243) : 1,
    (88,255,52): 2,
    (255,28,36): 3,
    (77,241,232): 4,
    (209,209,216): 5
}

colors = [(6,4,243), (88,255,52), (255,28,36), (77,241,232), (209,209,216)]

img = cv2.imread('Image Segmentation Data/segmented_2/27_FeMoS2_24_8.png', -1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def get_mask(segmented):
   for i in range(segmented.shape[0]):
      for j in range(segmented.shape[1]):
         segmented[i][j] = np.array([0,0,0]) if tuple(segmented[i][j]) not in colors else segmented[i][j]
   plt.imshow(img)
   plt.show()

get_mask(img)