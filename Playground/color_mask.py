from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import os
from matplotlib.patches import Rectangle

def img_is_color(img):

    if len(img.shape) == 3:
        # Check the color channels to see if they're all the same.
        c1, c2, c3 = img[:, : , 0], img[:, :, 1], img[:, :, 2]
        if (c1 == c2).all() and (c2 == c3).all():
            return True

    return False

def show_image_list(list_images, list_titles=None, list_cmaps=None, grid=True, num_cols=2, figsize=(20, 8), title_fontsize=30):
    '''
    Shows a grid of images, where each image is a Numpy array. The images can be either
    RGB or grayscale.

    Parameters:
    ----------
    images: list
        List of the images to be displayed.
    list_titles: list or None
        Optional list of titles to be shown for each image.
    list_cmaps: list or None
        Optional list of cmap values for each image. If None, then cmap will be
        automatically inferred.
    grid: boolean
        If True, show a grid over each image
    num_cols: int
        Number of columns to show.
    figsize: tuple of width, height
        Value to be passed to pyplot.figure()
    title_fontsize: int
        Value to be passed to set_title().
    '''

    assert isinstance(list_images, list)
    assert len(list_images) > 0
    assert isinstance(list_images[0], np.ndarray)

    if list_titles is not None:
        assert isinstance(list_titles, list)
        assert len(list_images) == len(list_titles), '%d imgs != %d titles' % (len(list_images), len(list_titles))

    if list_cmaps is not None:
        assert isinstance(list_cmaps, list)
        assert len(list_images) == len(list_cmaps), '%d imgs != %d cmaps' % (len(list_images), len(list_cmaps))

    num_images  = len(list_images)
    num_cols    = min(num_images, num_cols)
    num_rows    = int(num_images / num_cols) + (1 if num_images % num_cols != 0 else 0)

    # Create a grid of subplots.
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    
    # Create list of axes for easy iteration.
    if isinstance(axes, np.ndarray):
        list_axes = list(axes.flat)
    else:
        list_axes = [axes]

    for i in range(num_images):

        img    = list_images[i]
        title  = list_titles[i] if list_titles is not None else 'Image %d' % (i)
        cmap   = list_cmaps[i] if list_cmaps is not None else (None if img_is_color(img) else 'gray')
        
        list_axes[i].imshow(img, cmap=cmap)
        list_axes[i].set_title(title, fontsize=title_fontsize) 
        list_axes[i].grid(grid)

    for i in range(num_images, len(list_axes)):
        list_axes[i].set_visible(False)

    fig.tight_layout()
    _ = plt.show()

def normalize_img(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Sharpen using unsharp mask
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    sharpened_image = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
    # Normalize image intensities
    normalized_image = cv2.normalize(sharpened_image, None, 0, 255, cv2.NORM_MINMAX)
    return normalized_image



def create_folder(src_folder: str, dest_folder:str, technique):
    os.makedirs(f"../Image Segmentation Data/{dest_folder}", exist_ok=True)
    for image in os.scandir(f"../Image Segmentation Data/{src_folder}"):
        if image.name[-3:] == 'png':
            img = cv2.imread(image.path)
            after_image = Image.fromarray(technique(img))
            after_image.save(f'../Image Segmentation Data/{dest_folder}/' + image.name)


# img = cv2.imread(file_path)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# # Sharpen using unsharp mask
# blurred = cv2.GaussianBlur(img, (0, 0), 3)
# sharpened_image = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)

# # Normalize image intensities
# normalized_image = cv2.normalize(sharpened_image, None, 0, 255, cv2.NORM_MINMAX)

# mask = np.where(normalized_image[...,2] > normalized_image[...,0] * 1.0,1,0)

# show_image_list([normalized_image, mask], ["Normalized", "Mask"])

if __name__ == '__main__':
    create_folder('images_2', 'normalized_2', normalize_img)