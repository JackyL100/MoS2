import os
from PIL import Image
import cv2
from color_quantization import color_quant_kmeans
from contrast import contrast

def create_folder(src_folder: str, dest_folder:str, technique, **kwargs):
    """
    Creates a folder by processing images using a given technique

    Args:
        src_folder (str): folder containing images to be used
        dest_folder (str): folder where processed images will be saved
        technique (array of functions): function that processes image, may have multiple outputs but processed image should be last
        kwargs (dict): dictionary of kwargs for functions in technique
    """
    os.makedirs(f"Image Segmentation Data/{dest_folder}", exist_ok=True)
    formats = ['png', 'jpg']
    for image in os.scandir(f"Image Segmentation Data/{src_folder}"):
        if image.name[-3:] not in formats:
            continue
        original = cv2.imread(image.path)
        img = Image.new('RGB', (588, 780))
        for i,kwarg in kwargs.items():
            original = technique[int(i)](original, **kwarg)
            if isinstance(original, tuple):
                original = original[-1]
        img = Image.fromarray(original)
        img.save(f'Image Segmentation Data/{dest_folder}/' + image.name)

if __name__ == '__main__':
    kwarg = {
        '0' : {
            'alpha' : 1.7,
            'beta' : -40
        },
        '1' : {
            'n_colors' : 5,
            'denoise' : True,
            'BGR2RGB' : True
        }
    }
    create_folder('images_2', 'contrast_blue_quantized_5', [contrast, color_quant_kmeans], **kwarg)
