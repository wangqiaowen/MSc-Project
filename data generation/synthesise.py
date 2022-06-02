import sys
import os
import cv2
import numpy as np
import json
import SimpleITK as sitk
import pandas as pd
from tool import *
from drr import *
from utils import *



def synthesis(tool_dir, save_dir, lnd_dir, image_count, start_name=1, save_GT=False, save_seg_mask=True, verbose=True):
# The synthesis function created the simulated DRRs needed for network training
# As input this function takes the directory of the tool, the save directory, the total number of images\
# to be cretaed and the directory where the 3D landmark coordinates are stored
# As output, this function returns the DRR, the segmentation mask and the 2D landmark coordinates

    # The number of the first image to be saved
    st_num = int(start_name)  

    for i in range(1, image_count+1):
        if verbose and i==1:
            print('Loading the Tool volume and seting up the sysnthesis pipeline ...')

        tool_vol = sitk.ReadImage(tool_dir)
        lnd = np.array(pd.read_csv(lnd_dir, sep=',', header=None).values)[:, 0:3]
        lnd = lnd.astype(np.float)

        tool_obj = tool(tool_vol, lnd, verbose=False)

        # Setup up the range of transformation parameters. The DRRs will be created based on 
        # the tool that is transformed with parameters within this range
        tool_obj.SetTransformRange(trn_range=[-65, 65], rot_range=[-35, 35])
        # tool_obj.SetTransformationParameters(trn=[i/2, 0, i/2], rot=[i, 0, 0])
        tool_vol, trn_landmarks = tool_obj.GetTool(use_reference=True)


        if verbose:
            print('Now Synthesizing image', str(st_num), end='\r')

        drr_img, P, camera_matrix = CreateDRR(sitk2itk(tool_vol), threshold=0)

        image_points = ProjectObjectToImage(trn_landmarks, projection_matrix=P)
        
        output_image = sitk.GetArrayFromImage(itk2sitk(drr_img))[0, :, :]


        if save_GT:
            gt_image = PutPointsOnImage(output_image, points=image_points, name="pose3.png", show=False)
            cv2.imwrite(os.path.join(save_dir, str(st_num)+'_gt.png'), gt_image)
        
        if save_seg_mask:
            mask = (255-output_image)>0.1
            cv2.imwrite(os.path.join(save_dir, str(st_num)+'_mask.png'), mask*255)
            
        cv2.imwrite(os.path.join(save_dir, str(st_num)+'.png'), output_image)
        np.savetxt(os.path.join(save_dir, str(st_num)+'.txt'), image_points, delimiter=',')

        st_num += 1

    return 0



if __name__=="__main__":
    cf = open(sys.argv[1])
    data = json.load(cf)
    synthesis(tool_dir=data["Tool_DIR"], save_dir=data["Save_DIR"], lnd_dir=data["LND_DIR"],\
        image_count=data["IMAGE_COUNT"], start_name=data["START_NAME"])

