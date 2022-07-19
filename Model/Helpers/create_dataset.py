import numpy as np
from  matplotlib import pyplot as plt
import matplotlib.image as mpimg
import cv2
import shutil
import os

def create_dataset(n_imgs, img_folder, masks_folder, coor_folder, img_size, mask_size, coor_scaler):
    '''
    :param n_imgs: int
        how many imgs to process
    :param img_folder: str
        img folder path
    :param masks_folder: str
        masks folder path
    :param coor_folder: str
        coordinates folder path
    :param img_size: tuple
        the output img size e.g.(128, 128)
    :param mask_size: tuple
        the output mask size e.g.(64, 64)
    :param coor_scaler: int
        the scaler for coordinates, 
        e.g. if original img is in size (512, 512) and resized img is (256, 256) and resized mask is in size (128, 128), then this field should be 4

    ----------
    
    :return img_data_array : np.array
        resized images
    :return masks_array : np.array
        resized masks
    :return coor_array : np.array
        rescaled coordinates

    '''
    img_data_array = []
    masks_array = []
    coor_array = []
    j=0

    for i in range(1,n_imgs+1):
        img_path = os.path.join(img_folder, str(i)+".png")
        mask_path = os.path.join(masks_folder, str(i)+"_mask.png")
        coordinate_path = os.path.join(coor_folder, str(i)+".txt")

        image = cv2.imread(img_path)
        image = cv2.resize(image, (img_size, img_size))  ######
        image = np.array(image)
        image = image.astype('float32')
        # Normalization
        image /= 255 
        img_data_array.append(image)

        mask = cv2.imread(mask_path)
        mask = cv2.resize(mask, (mask_size, mask_size))
        mask = np.array(mask)
        mask = mask.astype('float32')
        # Normalization
        mask /= 255 
        masks_array.append(mask)

        with open(coordinate_path, 'r') as coor:
            new_coor = []
            for line in coor.readlines():
                coors = line.split(",")
                coors = [float(x) for x in coors]
                new_coor.extend(coors)
            coor_array.append(new_coor)

        masks_array_np = np.asarray(masks_array)
        img_data_array_np = np.asarray(img_data_array)

        # Remove unused channels
        masks_array_np = np.delete(masks_array_np, 1, axis=3)
        masks_array_np = np.delete(masks_array_np, 1, axis=3)

        # Rescale coordinates
        coor_array_np = np.asarray(coor_array)/coor_scaler

    return img_data_array_np, masks_array_np, coor_array_np
