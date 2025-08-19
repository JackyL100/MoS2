from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg
from PIL import Image, ImageFilter
import cv2

def compare_imgs(imgs, titles):
    fig, axarr = plt.subplots(2,2)
    fig.set_size_inches(13,9)
    fig.tight_layout()
    count = 0
    for i in range(2):
        for j in range(2):
            axarr[i][j].title.set_text(titles[count])
            axarr[i][j].imshow(imgs[count])
            axarr[i][j].axis('off')
            count += 1
    plt.show()

def color_quant_kmeans(img,n_colors, denoise=False):

    dim = img.shape
    vectorized_img = np.zeros(dim)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if denoise:
        # Sharpen using unsharp mask
        blurred = cv2.GaussianBlur(img, (0, 0), 3)
        sharpened_image = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)

        # Normalize image intensities
        normalized_image = cv2.normalize(sharpened_image, None, 0, 255, cv2.NORM_MINMAX)
        vectorized_img = np.array(normalized_image).reshape(-1,3)/255.0
    else:
        vectorized_img = img.reshape(-1,3)/255.0

    km = KMeans(n_clusters=n_colors, max_iter=100, tol=0.001, init='random', n_init=5)
    km.fit(vectorized_img[...,0].reshape(-1,1))
    

    Xin = km.labels_

    # The image in new colors
    cluster_centers = np.round(km.cluster_centers_,3)
    Xnew = cluster_centers[Xin,:]

    # Bring it back to original dimensions
    Xnim = np.zeros((dim[0], dim[1],3))
    Xnim[...,0] = Xnew.squeeze().reshape((588,780))
    Xnim[...,1] = vectorized_img[...,1].reshape((588,780))
    Xnim[...,2] = vectorized_img[...,2].reshape(588,780)
    Xnim *= 255.0
    Xnim = Xnim.astype(np.uint8)
    if denoise:
        return img, np.array(normalized_image), Xnim
    else:
        return img, None, Xnim

def get_difference_between_imgs(original, other_imgs, other_imgs_names, print_diff=True):
    assert len(other_imgs) == len(other_imgs_names)
    differences = []
    for i in range(len(other_imgs)):
        difference = mean_squared_error(np.array(original).flatten(), np.array(other_imgs[i]).flatten())
        if print_diff:
            print(f'Difference between original and {other_imgs_names[i]}: {difference}')
        differences.append(difference)
    return differences


def find_best_n_colors(file_path):
    differences = np.zeros(25)
    colors = np.linspace(5,29,25)

    for i in range(5,30):
        n_colors = i
        #original, denoised, quantized_denoised = color_quant_kmeans(file_path, n_colors, denoise=True)

        original,_,quantized = color_quant_kmeans(file_path,  n_colors, denoise=False)
        #compare_imgs([original, denoised, quantized_denoised, quantized], ["Original", "Denoised", "Quantized and Denoised","Quantized"])

        differences[i-5] = get_difference_between_imgs(original, [quantized],["Quantized"], print_diff=False)[0]

    plt.scatter(colors, differences)
    plt.title(file_path)
    plt.show()


def normalize_img(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Sharpen using unsharp mask
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    sharpened_image = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
    # Normalize image intensities
    normalized_image = cv2.normalize(sharpened_image, None, 0, 255, cv2.NORM_MINMAX)
    return normalized_image


def create_folder(src_folder: str, dest_folder:str, technique:str):
    available_tech = ['quantized', 'denoised', 'normalized']
    assert technique in available_tech
    os.makedirs(f"Image Segmentation Data/{dest_folder}", exist_ok=True)
    for image in os.scandir(f"Image Segmentation Data/{src_folder}"):
        # original = cv2.cvtColor(cv2.imread(image.path), cv2.COLOR_BGR2RGB)
        original = cv2.imread(image.path)
        if technique == 'quantized':
            _, _, quantized_img = color_quant_kmeans(original, 15, denoise=False)
            img = Image.fromarray(quantized_img)
            img.save(f'Image Segmentation Data/{dest_folder}/' + image.name)
        elif technique == 'denoised':
            _, denoised, _ = color_quant_kmeans(original, 15, denoise=True)
            img = Image.fromarray(denoised)
            img.save(f'Image Segmentation Data/{dest_folder}/' + image.name)
        elif technique == 'normalized':
            img = Image.fromarray(normalize_img(cv2.imread(image.path)))
            img.save(f'Image Segmentation Data/{dest_folder}/' + image.name)

if __name__ == '__main__':
    # n_colors = 15
    # file_path = 'Image Segmentation Data/normalized/14_FeMoS2_1.jpg'
    # img = cv2.imread(file_path)
    # original, denoised, quantized_denoised = color_quant_kmeans(img, n_colors, denoise=True)
    # _, _, quantized = color_quant_kmeans(img, n_colors, denoise=False)
    # compare_imgs([original, denoised, quantized_denoised, quantized], ["Original", "Denoised", "Quantized and Denoised","Quantized"])

    create_folder("normalized_2", "normalized_quantized_2", 'quantized')