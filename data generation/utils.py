import imp
import math
import itk
import SimpleITK as sitk
from itk.support.types import Matrix
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from scipy.spatial import distance_matrix
from sklearn.decomposition import PCA

from termcolor import colored

from widget2 import *


def sitk2itk(img):
    arr = sitk.GetArrayFromImage(img)
    itk_img = itk.GetImageFromArray(arr)
    itk_img.SetOrigin(img.GetOrigin())
    itk_img.SetDirection(itk.GetMatrixFromArray(
        np.reshape(np.array(img.GetDirection()), [3]*2)))
    itk_img.SetSpacing(img.GetSpacing())
    return itk_img.astype(itk.D)


def itk2sitk(img):
    arr = itk.GetArrayFromImage(img)
    sitk_img = sitk.GetImageFromArray(arr)
    sitk_img.SetOrigin(tuple(img.GetOrigin()))
    sitk_img.SetDirection(itk.GetArrayFromMatrix(img.GetDirection()).flatten())
    sitk_img.SetSpacing(tuple(img.GetSpacing()))
    return(sitk_img)


def PutPointsOnImage(img, points=None, name="image.png", show=True):

    img = img.astype('uint8')
    img_cv = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

    if points.all() != None:
        for i, (x, y) in enumerate(points):
            px = int(x)
            py = int(y)
            # cv.circle(img_cv, (py,px), 3, (255, 0, 0), 1)
            cv.circle(img_cv, (px, py), 3, (0, 255, 0), 1)
            cv.putText(img_cv, str(i+1), (px+10, py+10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    if show:
        cv.namedWindow(name, cv.WINDOW_NORMAL)
        cv.resizeWindow(name, 512, 512)
        cv.imshow(name, img_cv)
        cv.waitKey(0)
        cv.destroyAllWindows()
        cv.imwrite(name, img_cv)
    return(img_cv)


def EstimateObjectPose(img_points, obj_points, camera_matrix):
    dist_coefs = np.zeros((4, 1))
    success, rotation_vector, translation_vector = cv.solvePnP(obj_points, img_points,
                                                               cameraMatrix=camera_matrix, distCoeffs=dist_coefs, flags=0)

    return translation_vector, rotation_vector, camera_matrix, dist_coefs


def ProjectObjectToImage(object_points, projection_matrix):
    image_points = np.transpose(
        projection_matrix @ np.transpose(np.c_[object_points, np.ones(len(object_points))]))
    image_points = image_points / image_points[:, 2][:, None]
    return (image_points[0:, 0:2])


def createHeatMap(lnd, radius, input_size=(512, 512), output_size=(512, 512), show=False, num_point=6000):

    hm = np.zeros(output_size, np.float)
    lnd = np.asarray(lnd)
    lnd *= np.asarray(output_size) / np.asarray(input_size)
    if any(lnd >= output_size) or any(lnd < 0):
        return hm
    else:
        points = np.random.multivariate_normal(
            lnd, [[radius, 0], [0, radius]], num_point)
        for p in points:
            if any(p >= output_size) or any(p < 0):
                continue
            hm[int(p[1]), int(p[0])] += 1

        hm /= np.max(hm)
        hm = cv.GaussianBlur(hm, ksize=(11, 11), sigmaX=0)
    if show:
        name = 'how'
        cv.namedWindow(name, cv.WINDOW_NORMAL)
        cv.resizeWindow(name, 512, 512)
        cv.imshow(name, hm)
        cv.waitKey(0)
        cv.destroyAllWindows()

    return (hm)


def createHeatMap2(pt, sigma=100, output_size=[512, 512], input_size=[512, 512]):
    hm = np.zeros(output_size, np.float)

    pt *= np.asarray(output_size) / np.asarray(input_size)
    def scaledGaussian(x): return np.exp(-(1/(2*sigma))*(x**2))

    if any(pt >= output_size) or any(pt < 0):
        return hm
    else:
        th = 10
        for i in range(int(pt[0]-sigma/2-th), int(pt[0]+sigma/2+th)):
            for j in range(int(pt[1]-sigma/2-th), int(pt[1]+sigma/2+th)):
                if i < 0 or j < 0 or i >= output_size[0] or j >= output_size[1]:
                    continue

                distance = np.linalg.norm(np.array([i-pt[0], j-pt[1]]))
                scaledGaussianProb = scaledGaussian(distance)
                hm[j, i] = scaledGaussianProb
                hm[j, i] = np.clip(scaledGaussianProb*255, 0, 255)

    hm_blur = cv.blur(hm, (5, 5))
    hm_blur = hm_blur / np.max(hm_blur)
    hm_blur[hm_blur < 0.01] = 0
    return (hm_blur)


def colorHeatMap(img):
    hm_show = None
    hm_show = cv.normalize(img, hm_show, alpha=0, beta=255,
                           norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    hm_show = cv.applyColorMap(hm_show, cv.COLORMAP_JET)
    return hm_show


def rescale_to_eight_bit(img):
    max_inten = np.max(img)
    min_inten = np.min(img)

    outimg = (img - min_inten) / (max_inten-min_inten)

    return (outimg*255).astype(int)


def judge(img, coor_2d, rx, ry, rz, coor_2d_temp, trnMat_temp, img_temp, trnMat, output_image):
    outliers = np.any((coor_2d > 512) | (coor_2d < 0), axis=1)
#   show_img(img, coor_2d, coor_2d)
    # print(outliers)

    print(colored("rx: {:2f}, ry: {:2f}, rz: {:2f}".format(rx, ry, rz), "blue"))
    print(colored("tx: {:2f}, ty: {:2f}, tz: {:2f}".format(trnMat_temp[0], trnMat_temp[1], trnMat_temp[2]), "blue"))
    print(trnMat)

    pca = PCA()
    # temp = []
    # for i in range(512):
    #         for j in range(512):
    #             if(img[i][j] != 255):
    #                 data_point = [i, j]
    #                 temp.append(data_point)
    data_points = np.where(img != 255)
    data_points_temp = np.where(img_temp != 255)
    if data_points:

            data_points = np.transpose(np.array(data_points))
            pca.fit(data_points)
            explained_variance_ratio = pca.explained_variance_ratio_

            data_points_temp = np.transpose(np.array(data_points_temp))
            pca.fit(data_points_temp)
            explained_variance_ratio_temp = pca.explained_variance_ratio_

            dm = distance_matrix(coor_2d, coor_2d)

            contour_dis = np.sum(dm[0, 1]+dm[0, 2]+dm[1,5]+dm[2,5])
            dis = np.sum(dm)/2
            contribution_ratio = explained_variance_ratio[0] / explained_variance_ratio[1]
            dis_contribution_ratio = dis/(contribution_ratio*100)
            contour_and_all = dis + dm[0,1] + dm[0,5] + dm[1,5] + dm[2,5] + dm[0,2]
            dis_ratio_ratio = contour_and_all/(contribution_ratio*100)

            dm_temp = distance_matrix(coor_2d_temp, coor_2d_temp)
            contour_dis_temp = np.sum(dm_temp[0, 1]+dm_temp[0, 2]+dm_temp[1,5]+dm_temp[2,5])
            dis_temp = np.sum(dm_temp)/2
            contribution_ratio_temp = explained_variance_ratio_temp[0] / explained_variance_ratio_temp[1]
            dis_contribution_ratio_temp = dis_temp/(contribution_ratio_temp*100)
            contour_and_all_temp = dis_temp + dm_temp[0,1] + dm_temp[0,5] + dm_temp[1,5] + dm_temp[2,5] + dm_temp[0,2]
            new_dis_ratio_ratio = contour_and_all_temp/(contribution_ratio_temp*100)


            Dict = dict({
                'trn' : trnMat[:, -1][0:3],
                'rot' : [rx, ry, rz],
                'original': {
                    'dm' : dm,
                    'explained_variance_ratio' : explained_variance_ratio,
                    'contour_dis' : contour_dis,
                    'dis' : dis,
                    'contribution_ratio' : contribution_ratio,
                    'dis_contribution_ratio' : dis_contribution_ratio,
                    'contour_and_all' : contour_and_all,
                    'dis_ratio_ratio' : dis_ratio_ratio
                },
                'no_z_translation' : {
                    'dm' : dm_temp,
                    'explained_variance_ratio' : explained_variance_ratio_temp,
                    'contour_dis' : contour_dis_temp,
                    'dis' : dis_temp,
                    'contribution_ratio' : contribution_ratio_temp,
                    'dis_contribution_ratio' : dis_contribution_ratio_temp,
                    'contour_and_all' : contour_and_all_temp,
                    'dis_ratio_ratio' : new_dis_ratio_ratio
                }
            })


            if (outliers == True).any():
                print(colored("outside! reject","red"))
                reason = "outside"
                return False, reason, Dict


            # if(contour_dis <= 400 or contribution_ratio >= 3 or dis_contribution_ratio <= 4):
            # if(dis_contribution_ratio_temp <= 4):
            #     print(colored("reject: dis_contribution_ratio {:2f}, contribution_ratio {:2f}, contour_dis {:2f}, and dis {:2f}".format(
            #         dis_contribution_ratio, contribution_ratio, contour_dis, dis), "red"))
            #     print(colored("coorsponding to tz=0: dis_contribution_ratio_temp {:2f}, contribution_ratio {:2f}, contour_dis {:2f}, and dis {:2f}".format(
            #         dis_contribution_ratio_temp, contribution_ratio_temp, contour_dis_temp, dis_temp), "red"))
            #     return False
            # else:
            #     print(colored("accept: dis_contribution_ratio {:2f}, contribution_ratio {:2f}, contour_dis {:2f}, and dis {:2f}".format(
            #         dis_contribution_ratio, contribution_ratio, contour_dis, dis),"green"))
            #     print(colored("coorsponding to tz=0: dis_contribution_ratio_temp {:2f}, contribution_ratio {:2f}, contour_dis {:2f}, and dis {:2f}".format(
            #         dis_contribution_ratio_temp, contribution_ratio_temp, contour_dis_temp, dis_temp), "green"))
            #     return True

            if(dis_contribution_ratio_temp > 4 and new_dis_ratio_ratio >=8):
                print(colored("accept: dis_contribution_ratio {:2f}, contribution_ratio {:2f}, contour_dis {:2f}, dis {:2f}".format(
                    dis_contribution_ratio, contribution_ratio, contour_dis, dis),"green"))
                print(colored("coorsponding to tz=0: dis_contribution_ratio_temp {:2f}, contribution_ratio {:2f}, contour_dis {:2f}, and dis {:2f} contour_and_all {:2f}, new_dis_ratio_ratio {:2f}".format(
                    dis_contribution_ratio_temp, contribution_ratio_temp, contour_dis_temp, dis_temp, contour_and_all, new_dis_ratio_ratio),"green"))
                reason = "comply"
                return True, reason, Dict

            else:

                print(colored("reject: dis_contribution_ratio {:2f}, contribution_ratio {:2f}, contour_dis {:2f}, dis {:2f} ".format(
                    dis_contribution_ratio, contribution_ratio, contour_dis, dis),"red"))
                print(colored("coorsponding to tz=0: dis_contribution_ratio_temp {:2f}, contribution_ratio {:2f}, contour_dis {:2f}, dis {:2f} contour_and_all {:2f}, new_dis_ratio_ratio {:2f}".format(
                    dis_contribution_ratio_temp, contribution_ratio_temp, contour_dis_temp, dis_temp, contour_and_all, new_dis_ratio_ratio),"red"))
                reason = "disobey"
                return False, reason, Dict
            


    else:
            print(colored("reject: empty img"),"yellow")
            reason = "empty"
            return False, reason, Dict

def judge2(img, coor_2d, rx, ry, rz, coor_2d_temp, trnMat_temp, img_temp, trnMat, output_image):
    global human_judgement
    outliers = np.any((coor_2d > 512) | (coor_2d < 0), axis=1)
#   show_img(img, coor_2d, coor_2d)
    # print(outliers)

    print(colored("rx: {:2f}, ry: {:2f}, rz: {:2f}".format(rx, ry, rz), "blue"))
    print(colored("tx: {:2f}, ty: {:2f}, tz: {:2f}".format(trnMat_temp[0], trnMat_temp[1], trnMat_temp[2]), "blue"))
    print(trnMat)


    # if (outliers == True).any():
    #         print(colored("outside! reject","red", attrs=['bold']))
    #         reason = "outside"
    #         return False, reason

    pca = PCA()
    # temp = []
    # for i in range(512):
    #         for j in range(512):
    #             if(img[i][j] != 255):
    #                 data_point = [i, j]
    #                 temp.append(data_point)
    data_points = np.where(img != 255)
    data_points_temp = np.where(img_temp != 255)
    if data_points:

            data_points = np.transpose(np.array(data_points))
            pca.fit(data_points)
            explained_variance_ratio = pca.explained_variance_ratio_

            data_points_temp = np.transpose(np.array(data_points_temp))
            pca.fit(data_points_temp)
            explained_variance_ratio_temp = pca.explained_variance_ratio_

            dm = distance_matrix(coor_2d, coor_2d)

            contour_dis = np.sum(dm[0, 1]+dm[0, 2]+dm[1,5]+dm[2,5])
            dis = np.sum(dm)/2
            contribution_ratio = explained_variance_ratio[0] / explained_variance_ratio[1]
            dis_contribution_ratio = dis/(contribution_ratio*100)
            contour_and_all = dis + dm[0,1] + dm[0,5] + dm[1,5] + dm[2,5] + dm[0,2]
            dis_ratio_ratio = contour_and_all/(contribution_ratio*100)

            dm_temp = distance_matrix(coor_2d_temp, coor_2d_temp)
            contour_dis_temp = np.sum(dm_temp[0, 1]+dm_temp[0, 2]+dm_temp[1,5]+dm_temp[2,5])
            dis_temp = np.sum(dm_temp)/2
            contribution_ratio_temp = explained_variance_ratio_temp[0] / explained_variance_ratio_temp[1]
            dis_contribution_ratio_temp = dis_temp/(contribution_ratio_temp*100)
            contour_and_all_temp = dis_temp + dm_temp[0,1] + dm_temp[0,5] + dm_temp[1,5] + dm_temp[2,5] + dm_temp[0,2]
            new_dis_ratio_ratio = contour_and_all_temp/(contribution_ratio_temp*100)


            Dict = dict({
                'trn' : trnMat[:, -1][0:3],
                'rot' : [rx, ry, rz],
                'original': {
                    'dm' : dm,
                    'explained_variance_ratio' : explained_variance_ratio,
                    'contour_dis' : contour_dis,
                    'dis' : dis,
                    'contribution_ratio' : contribution_ratio,
                    'dis_contribution_ratio' : dis_contribution_ratio,
                    'contour_and_all' : contour_and_all,
                    'dis_ratio_ratio' : dis_ratio_ratio
                },
                'no_z_translation' : {
                    'dm' : dm_temp,
                    'explained_variance_ratio' : explained_variance_ratio_temp,
                    'contour_dis' : contour_dis_temp,
                    'dis' : dis_temp,
                    'contribution_ratio' : contribution_ratio_temp,
                    'dis_contribution_ratio' : dis_contribution_ratio_temp,
                    'contour_and_all' : contour_and_all_temp,
                    'dis_ratio_ratio' : new_dis_ratio_ratio
                }
            })


            if (outliers == True).any():
                print(colored("outside! reject","red", attrs=['bold']))
                reason = "outside"
                return False, reason, Dict

            # cv2.imwrite('img.png', output_image)
            image = cv2.imread('./temp/'+'temp.png')
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            thresh = 254.9999
            ret,thresh_img = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)
            thresh_img = thresh_img.astype(np.uint8)
            contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            if(len(contours)>=3):
                reason = "more than 2 contours"
                return True, reason, Dict

            # contours_length = 0
            # for i in range(1, len(contours)):
            #     contours_length += cv2.arcLength(contours[i], True)


            # if(contour_dis <= 400 or contribution_ratio >= 3 or dis_contribution_ratio <= 4):
            # if(dis_contribution_ratio_temp <= 4):
            #     print(colored("reject: dis_contribution_ratio {:2f}, contribution_ratio {:2f}, contour_dis {:2f}, and dis {:2f}".format(
            #         dis_contribution_ratio, contribution_ratio, contour_dis, dis), "red"))
            #     print(colored("coorsponding to tz=0: dis_contribution_ratio_temp {:2f}, contribution_ratio {:2f}, contour_dis {:2f}, and dis {:2f}".format(
            #         dis_contribution_ratio_temp, contribution_ratio_temp, contour_dis_temp, dis_temp), "red"))
            #     return False
            # else:
            #     print(colored("accept: dis_contribution_ratio {:2f}, contribution_ratio {:2f}, contour_dis {:2f}, and dis {:2f}".format(
            #         dis_contribution_ratio, contribution_ratio, contour_dis, dis),"green"))
            #     print(colored("coorsponding to tz=0: dis_contribution_ratio_temp {:2f}, contribution_ratio {:2f}, contour_dis {:2f}, and dis {:2f}".format(
            #         dis_contribution_ratio_temp, contribution_ratio_temp, contour_dis_temp, dis_temp), "green"))
            #     return True
            x_four_two = coor_2d_temp[3][1]
            x_six_one = coor_2d_temp[5][0]
            x_three_one = coor_2d_temp[2][0]
            x_six_two = coor_2d_temp[5][1]
            x_three_two = coor_2d_temp[2][1]
            x_one_two = coor_2d_temp[0][1]

            print(colored("distances: x_four_two {:2f}, x_six_one {:2f}, x_three_one {:2f}, x_six_two {:2f},  x_three_two {:2f}".format(
                            x_four_two, x_six_one, x_three_one, x_six_two, x_three_two),"grey"))


            if(True):

                if(dis_contribution_ratio_temp > 4 and new_dis_ratio_ratio >=8 and x_three_one > 60 and x_one_two > 41 and x_six_one >= 80):
                    print(colored("accept: dis_contribution_ratio {:2f}, contribution_ratio {:2f}, contour_dis {:2f}, dis {:2f}".format(
                        dis_contribution_ratio, contribution_ratio, contour_dis, dis),"green"))
                    print(colored("coorsponding to tz=0: dis_contribution_ratio_temp {:2f}, contribution_ratio {:2f}, contour_dis {:2f}, and dis {:2f} contour_and_all {:2f}, new_dis_ratio_ratio {:2f}".format(
                        dis_contribution_ratio_temp, contribution_ratio_temp, contour_dis_temp, dis_temp, contour_and_all, new_dis_ratio_ratio),"green"))
                    reason = "comply"
                    return True, reason, Dict

                elif(new_dis_ratio_ratio >= 3 and new_dis_ratio_ratio < 8.5):
                    human_judgement = show_window(output_image)
                    if(human_judgement):
                        print(colored("accept: dis_contribution_ratio {:2f}, contribution_ratio {:2f}, contour_dis {:2f}, dis {:2f}".format(
                            dis_contribution_ratio, contribution_ratio, contour_dis, dis),"green"))
                        print(colored("coorsponding to tz=0: dis_contribution_ratio_temp {:2f}, contribution_ratio {:2f}, contour_dis {:2f}, and dis {:2f} contour_and_all {:2f}, new_dis_ratio_ratio {:2f}".format(
                            dis_contribution_ratio_temp, contribution_ratio_temp, contour_dis_temp, dis_temp, contour_and_all, new_dis_ratio_ratio),"green"))
                        reason = "human accept"
                        return True, reason, Dict
                    else:
                        print(colored("reject: dis_contribution_ratio {:2f}, contribution_ratio {:2f}, contour_dis {:2f}, dis {:2f} ".format(
                            dis_contribution_ratio, contribution_ratio, contour_dis, dis),"red"))
                        print(colored("coorsponding to tz=0: dis_contribution_ratio_temp {:2f}, contribution_ratio {:2f}, contour_dis {:2f}, dis {:2f} contour_and_all {:2f}, new_dis_ratio_ratio {:2f}".format(
                            dis_contribution_ratio_temp, contribution_ratio_temp, contour_dis_temp, dis_temp, contour_and_all, new_dis_ratio_ratio),"red"))
                        reason = "human reject"
                        return False, reason, Dict

                else:

                    print(colored("reject: dis_contribution_ratio {:2f}, contribution_ratio {:2f}, contour_dis {:2f}, dis {:2f} ".format(
                        dis_contribution_ratio, contribution_ratio, contour_dis, dis),"red"))
                    print(colored("coorsponding to tz=0: dis_contribution_ratio_temp {:2f}, contribution_ratio {:2f}, contour_dis {:2f}, dis {:2f} contour_and_all {:2f}, new_dis_ratio_ratio {:2f}".format(
                        dis_contribution_ratio_temp, contribution_ratio_temp, contour_dis_temp, dis_temp, contour_and_all, new_dis_ratio_ratio),"red"))
                    reason = "disobey"
                    return False, reason, Dict
            
            # else:
            #     print(colored("reject: distances not satisfied","yellow", attrs=['bold']))
            #     reason = "distances"
            #     return False, reason
        

    else:
            print(colored("reject: empty img","yellow", attrs=['bold']))
            reason = "empty"
            return False, reason, Dict

