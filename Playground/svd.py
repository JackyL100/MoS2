import os
import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def display_image(arr_image,color_map=None,dim=(10,10)):
    arr_image = arr_image/np.max(arr_image)
    arr_image = np.uint8(arr_image*255)
    plt.figure(figsize=dim)
    plt.axis('off')
    plt.imshow(arr_image, cmap=color_map)
    plt.show()

def low_rank_reconstruction(path, k):
    f = os.path.join("",path)
    if f[-3:] != "jpg":
        return None
    if os.path.isfile(f):
        pix = mpimg.imread(f)
        fig, axarr = plt.subplots(1,2)
        if pix.shape[2] > 3:
            pix = pix[:,:,:3]
        pix_rgb = pix/255.0
        U0,S0,V0 = np.linalg.svd(pix_rgb[...,0])
        U1,S1,V1 = np.linalg.svd(pix_rgb[...,1])
        U2,S2,V2 = np.linalg.svd(pix_rgb[...,2])
        pix_compressed = np.zeros_like(pix_rgb)
        
        pix_compressed[...,0] = np.dot(U0[:,:k], V0[:k,:]*S0[:k].reshape(k,1))
        pix_compressed[...,1] = np.dot(U1[:,:k], V1[:k,:]*S1[:k].reshape(k,1))
        pix_compressed[...,2] = np.dot(U2[:,:k], V2[:k,:]*S2[:k].reshape(k,1))
        axarr[0].imshow(pix)
        axarr[0].title.set_text("Original")
        axarr[1].imshow(pix_compressed)
        axarr[0].axis('off')
        axarr[1].axis('off')
        axarr[1].title.set_text("Low Rank Approximation")
        fig.tight_layout()
        plt.show()
        return None
    else:
        return None
    
low_rank_reconstruction('Image Segmentation Data/images/9_FeMoS2_2_50x.jpg',100)