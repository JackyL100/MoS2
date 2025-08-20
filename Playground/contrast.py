import numpy as np
import cv2
import matplotlib.pyplot as plt
from color_quantization import color_quant_kmeans

def contrast(image, alpha, beta):
    new_image = np.zeros(image.shape, dtype=image.dtype)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            for c in range(image.shape[2]):
                new_image[y,x,c] = np.clip(alpha*image[y,x,c] + beta, 0, 255)
    return new_image

if __name__ == '__main__':
    fig, axs = plt.subplots(1, 2)
    fig.tight_layout()
    img = cv2.imread('Image Segmentation Data/images_2/25_FeMoS2_5.png')
    if img is None:
        print('Could not open image')
        exit(0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    contrast_img = contrast(img, 1.7, -40)
    _, _ , contrast_quantized = color_quant_kmeans(contrast_img, 5, denoise=True, channel=2, BGR2RGB=False)
    _, _ , quantized = color_quant_kmeans(img, 5, denoise=True, channel=2, BGR2RGB=False)
    axs[0].set_title("Contrast Quantized Image")
    axs[0].imshow(contrast_quantized)
    axs[1].set_title("Quantized Image")
    axs[1].imshow(quantized)
    plt.show()