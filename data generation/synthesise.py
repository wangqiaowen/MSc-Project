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
import tkinter as tk
import copy
import json


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def synthesis(tool_dir, save_dir, lnd_dir, image_count, garbage_dir, start_name=1, save_GT=False, save_seg_mask=True, verbose=True):
    # The synthesis function created the simulated DRRs needed for network training
    # As input this function takes the directory of the tool, the save directory, the total number of images\
    # to be created and the directory where the 3D landmark coordinates are stored
    # As output, this function returns the DRR, the segmentation mask and the 2D landmark coordinates

    # The number of the first image to be saved
    st_num = int(start_name)
    garbage_num = int(start_name)

    for i in range(1, image_count+1):
        if verbose and i == 1:
            print('Loading the Tool volume and seting up the sysnthesis pipeline ...')

        tool_vol = sitk.ReadImage(tool_dir)
        tool_vol_temp = sitk.ReadImage(tool_dir)

        # GUESS: lnd.shape = (6,3)
        lnd = np.array(pd.read_csv(lnd_dir, sep=',',
                       header=None).values)[:, 0:3]

        lnd = lnd.astype(np.float)

        tool_obj = tool(tool_vol, lnd, verbose=False)

        tool_obj_temp = tool(tool_vol_temp, lnd, verbose=False)

        # Setup up the range of transformation parameters. The DRRs will be created based on
        # the tool that is transformed with parameters within this range
        tool_obj.SetTransformRange(trn_range=[-90, 90], rot_range=[-180, 180])
        # tool_obj.SetTransformationParameters(trn=[i/2, 0, i/2], rot=[i, 0, 0])
        tool_vol, trn_landmarks, rx, ry, rz, trnMat = tool_obj.GetTool(use_reference=True)
        # trn_landmarks: transformed 3D landmark coordinates

        # Generate a temporary tool which has exactly the same rotation and translation except the translation along z axis.
        # The translation along z-axis is set to 0.
        trn_temp = copy.deepcopy(trnMat[:, -1][0:3])
        trn_temp[-1] = 0
        tool_obj_temp.SetTransformationParameters(trn_temp, [rx, ry, rz])
        tool_vol_temp, trn_landmarks_temp, rx_temp, ry_temp, rz_temp, trnMat_temp = tool_obj_temp.GetTool(use_reference=True)

        if verbose:
            print('Now Synthesizing image', str(st_num), end='\r')

        # P: Projection Matrix (3,4)
        # drr_img: drr_img.shape = (1, 512, 512)
        # camera_matrix: camera_matrix.shape = (3, 3)
        drr_img, P, camera_matrix = CreateDRR(sitk2itk(tool_vol), threshold=0)
        drr_img_temp, P_temp, camera_matrix_temp = CreateDRR(sitk2itk(tool_vol_temp), threshold=0)

        image_points = ProjectObjectToImage(trn_landmarks, projection_matrix=P)
        image_points_temp = ProjectObjectToImage(trn_landmarks_temp, projection_matrix=P_temp)
        # outliers = np.any((image_points > 512) | (image_points < 0), axis=1)
        # image_points[outliers] = [0,0]

        # (1, 512, 512) -> (512, 512)
        output_image = sitk.GetArrayFromImage(itk2sitk(drr_img))[0, :, :]
        output_image_temp = sitk.GetArrayFromImage(
            itk2sitk(drr_img_temp))[0, :, :]

        accept, reason, dm_temp= judge2(output_image, image_points, rx, ry, rz, image_points_temp, trn_temp, output_image_temp, trnMat, output_image)

        if save_GT:
            gt_image = PutPointsOnImage(
                output_image, points=image_points, name="pose3.png", show=False)
            cv2.imwrite(os.path.join(
                save_dir, str(st_num)+'_gt.png'), gt_image)

        # if save_seg_mask:
        #     if accept:
        #         mask = (255-output_image) > 0.1
        #         cv2.imwrite(os.path.join(
        #             save_dir, str(st_num)+'_mask.png'), mask*255)

        if accept:
            if save_seg_mask:
                mask = (255-output_image) > 0.1
                cv2.imwrite(os.path.join(
                    save_dir, str(st_num)+'_mask.png'), mask*255)
            cv2.imwrite(os.path.join(
                save_dir, str(st_num)+'.png'), output_image)
            np.savetxt(os.path.join(save_dir, str(st_num)+'.txt'),
                       image_points, delimiter=',')
            np.save(os.path.join(save_dir, str(st_num)+'_dict.npy'),
                       dm_temp)
            json.dump( dm_temp, open( os.path.join(save_dir, str(st_num)+'_dict.json'), 'w' ), cls=NumpyEncoder )
            print(colored("img {:f} saved".format(st_num), "cyan", attrs=['bold']))

            st_num += 1

        else:
                if save_seg_mask:
                    mask = (255-output_image) > 0.1
                    cv2.imwrite(os.path.join(
                        garbage_dir, str(garbage_num)+'_mask.png'), mask*255)
                np.savetxt(os.path.join(garbage_dir, str(garbage_num)+'.txt'),
                        image_points, delimiter=',')
                cv2.imwrite(os.path.join(garbage_dir, str(
                    garbage_num)+'.png'), output_image)
                np.save(os.path.join(garbage_dir, str(garbage_num)+'_dict.npy'),
                        dm_temp)
                json.dump(dm_temp, open(os.path.join(garbage_dir, str(garbage_num)+'_dict.json'), 'w' ), cls=NumpyEncoder)
                print(colored("garbage {:f} saved".format(garbage_num), "magenta", attrs=['bold']))
                garbage_num += 1

    return 0


def synthesis_one(tool_dir, save_dir, lnd_dir, garbage_dir, start_name, save_GT=False, save_seg_mask=True, verbose=True):
    # The synthesis function created the simulated DRRs needed for network training
    # As input this function takes the directory of the tool, the save directory, the total number of images\
    # to be created and the directory where the 3D landmark coordinates are stored
    # As output, this function returns the DRR, the segmentation mask and the 2D landmark coordinates

    # The number of the first image to be saved
    st_num = int(start_name)

    if verbose and start_name == 1:
        print('Loading the Tool volume and seting up the sysnthesis pipeline ...')

    tool_vol = sitk.ReadImage(tool_dir)
    tool_vol_temp = sitk.ReadImage(tool_dir)

    # GUESS: lnd.shape = (6,3)
    lnd = np.array(pd.read_csv(lnd_dir, sep=',', header=None).values)[:, 0:3]

    lnd = lnd.astype(np.float)

    tool_obj = tool(tool_vol, lnd, verbose=False)

    tool_obj_temp = tool(tool_vol_temp, lnd, verbose=False)

    # Setup up the range of transformation parameters. The DRRs will be created based on
    # the tool that is transformed with parameters within this range
    tool_obj.SetTransformRange(trn_range=[-90, 90], rot_range=[-180, 180])
    # tool_obj.SetTransformationParameters(trn=[i/2, 0, i/2], rot=[i, 0, 0])
    tool_vol, trn_landmarks, rx, ry, rz, trnMat = tool_obj.GetTool(use_reference=True)
    # trn_landmarks: transformed 3D landmark coordinates

    # Generate a temporary tool which has exactly the same rotation and translation except the translation along z axis.
    # The translation along z-axis is set to 0.
    trn_temp = copy.deepcopy(trnMat[:, -1][0:3])
    trn_temp[-1] = 0
    tool_obj_temp.SetTransformationParameters(trn_temp, [rx, ry, rz])
    tool_vol_temp, trn_landmarks_temp, rx_temp, ry_temp, rz_temp, trnMat_temp = tool_obj_temp.GetTool(use_reference=True)

    if verbose:
        print('Now Synthesizing image', str(st_num), end='\r')

        # P: Projection Matrix (3,4)
        # drr_img: drr_img.shape = (1, 512, 512)
        # camera_matrix: camera_matrix.shape = (3, 3)
    drr_img, P, camera_matrix = CreateDRR(sitk2itk(tool_vol), threshold=0)
    drr_img_temp, P_temp, camera_matrix_temp = CreateDRR(sitk2itk(tool_vol_temp), threshold=0)

    image_points = ProjectObjectToImage(trn_landmarks, projection_matrix=P)
    image_points_temp = ProjectObjectToImage(trn_landmarks_temp, projection_matrix=P_temp)
        # outliers = np.any((image_points > 512) | (image_points < 0), axis=1)
        # image_points[outliers] = [0,0]

        # (1, 512, 512) -> (512, 512)
    output_image = sitk.GetArrayFromImage(itk2sitk(drr_img))[0, :, :]
    output_image_temp = sitk.GetArrayFromImage(itk2sitk(drr_img_temp))[0, :, :]

    return output_image, image_points, rx, ry, rz, image_points_temp, trn_temp, output_image_temp, trnMat


def save(accept, st_num, garbage_dir, save_dir, output_image , image_points, garbage_num, dm_temp, save_seg_mask=True):

    # if save_seg_mask:
    #         if accept:
    #             mask = (255-output_image) > 0.1
    #             cv2.imwrite(os.path.join(
    #                 save_dir, str(num)+'_mask.png'), mask*255)

    # if accept:
    #         cv2.imwrite(os.path.join(
    #             save_dir, str(num)+'.png'), output_image)
    #         np.savetxt(os.path.join(save_dir, str(num)+'.txt'),
    #                    image_points, delimiter=',')

    # else:
    #         cv2.imwrite(os.path.join(garbage_dir, str(
    #             num)+'.png'), output_image)
    #         print(colored("garbage {:f} saved".format(num), "magenta"))

    if accept:
            if save_seg_mask:
                mask = (255-output_image) > 0.1
                cv2.imwrite(os.path.join(
                    save_dir, str(st_num)+'_mask.png'), mask*255)
            cv2.imwrite(os.path.join(
                save_dir, str(st_num)+'.png'), output_image)
            np.savetxt(os.path.join(save_dir, str(st_num)+'.txt'),
                       image_points, delimiter=',')
            np.save(os.path.join(save_dir, str(st_num)+'_dict.npy'),
                       dm_temp)
            json.dump( dm_temp, open( os.path.join(save_dir, str(st_num)+'_dict.json'), 'w' ), cls=NumpyEncoder )
            print(colored("img {:f} saved".format(st_num), "cyan", attrs=['bold']))

            st_num += 1

    else:
            if save_seg_mask:
                mask = (255-output_image) > 0.1
                cv2.imwrite(os.path.join(
                    garbage_dir, str(garbage_num)+'_mask.png'), mask*255)
            np.savetxt(os.path.join(garbage_dir, str(garbage_num)+'.txt'),
                       image_points, delimiter=',')
            cv2.imwrite(os.path.join(garbage_dir, str(
                garbage_num)+'.png'), output_image)
            np.save(os.path.join(garbage_dir, str(garbage_num)+'_dict.npy'),
                       dm_temp)
            json.dump(dm_temp, open(os.path.join(garbage_dir, str(garbage_num)+'_dict.json'), 'w' ), cls=NumpyEncoder)
            print(colored("garbage {:f} saved".format(garbage_num), "magenta", attrs=['bold']))
            garbage_num += 1

    return 0


if __name__ == "__main__":
    cf = open(sys.argv[1])
    data = json.load(cf)
    synthesis(tool_dir=data["Tool_DIR"], save_dir=data["Save_DIR"], lnd_dir=data["LND_DIR"],
              image_count=data["IMAGE_COUNT"], garbage_dir=data["Garbage_DIR"],  start_name=data["START_NAME"])

    # window = tk.Tk()
    # window.mainloop()
