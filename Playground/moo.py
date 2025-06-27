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

img = cv2.imread('Image Segmentation Data/denoised/4_MoS2_4.jpg', -1)

rgb_planes = cv2.split(img)

result_planes = []
result_norm_planes = []
for plane in rgb_planes:
    dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 21)
    diff_img = 255 - cv2.absdiff(plane, bg_img)
    norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    result_planes.append(diff_img)
    result_norm_planes.append(norm_img)
    
result = cv2.merge(result_planes)
result_norm = cv2.merge(result_norm_planes)

def adjust_gamma(image, gamma=1.0):

   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")

   return cv2.LUT(image, table)


gamma = 0.1                                 # change the value here to get different result
adjusted = adjust_gamma(result_norm, gamma=gamma)

plt.imshow(adjusted)
plt.show()