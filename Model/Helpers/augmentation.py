from random import random
import albumentations as A
import numpy as np


def transform(image, heatmaps, mask, coordinates):

    '''
    :params image: shape (height, weight, channel)
    :params heatmaps: shape (6, height, weight)
    :params mask: shape (height, weight, channel)
    :params coordinates: shape (6,2) , ATTENTION: the value of the coordinates


    :return result: dict
                    transformed img, heatmaps, mask, coordinates;
                    result['image'] : shape (height, weight, channel)
                    result['keypoints'] : shape (6,2)
                    result['mask'] : shape (height, weight, channel)
                    result['hms'] : shape (6, height, weight)
    '''

    transform = A.Compose([
            A.OneOf([
                A.Rotate(p=0.5),
                A.Flip(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.Transpose(p=0.5),
            ], p=1),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0, rotate_limit=45, p=0.5),
            ],

            keypoint_params=A.KeypointParams(format="xy"),
            additional_targets={'hm1': 'image', 'hm2': 'image', 
                                'hm3': 'image','hm4': 'image',
                                'hm5': 'image','hm6': 'image'},
            )
    random.seed(42)
    np.random.seed(42)

    h1 = heatmaps[0, : ,:]
    h2 = heatmaps[1, : ,:]
    h3 = heatmaps[2, : ,:]
    h4 = heatmaps[3, : ,:]
    h5 = heatmaps[4, : ,:]
    h6 = heatmaps[5, : ,:]

    transformed = transform(image=image, 
                           hm1 = h1,
                           hm2 = h2,
                           hm3 = h3,
                           hm4 = h4,
                           hm5 = h5,
                           hm6 = h6,
                           mask = mask, 
                           keypoints=coordinates)

    transformed_hm1 = transformed['hm1'] 
    transformed_hm2 = transformed['hm2']
    transformed_hm3 = transformed['hm3']
    transformed_hm4 = transformed['hm4']
    transformed_hm5 = transformed['hm5']
    transformed_hm6 = transformed['hm6']

    transformed_hms = [transformed_hm1, transformed_hm2, transformed_hm3, transformed_hm4, transformed_hm5, transformed_hm6]

    result = {}

    result['image'] = np.asarray(transformed['image'])
    result['keypoints'] = np.asarray(transformed['keypoints'])
    result['mask'] = np.asarray(transformed['mask'])
    result['hms'] = np.asarray(transformed_hms)

    return result

