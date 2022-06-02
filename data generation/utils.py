import itk 
import SimpleITK as sitk
from itk.support.types import Matrix
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt



def sitk2itk(img):
    arr = sitk.GetArrayFromImage(img)
    itk_img = itk.GetImageFromArray(arr)
    itk_img.SetOrigin(img.GetOrigin())
    itk_img.SetDirection(itk.GetMatrixFromArray(np.reshape(np.array(img.GetDirection()), [3]*2)))
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
            cv.circle(img_cv, (px,py), 3, (0, 255, 0), 1)
            cv.putText(img_cv, str(i+1), (px+10, py+10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    if show:
        cv.namedWindow(name, cv.WINDOW_NORMAL)
        cv.resizeWindow(name, 512, 512)
        cv.imshow(name, img_cv)
        cv.waitKey(0)
        cv.destroyAllWindows()
        cv.imwrite(name, img_cv)
    return(img_cv)


def EstimateObjectPose(img_points, obj_points, camera_matrix):
    dist_coefs = np.zeros((4,1))
    success, rotation_vector, translation_vector = cv.solvePnP(obj_points, img_points,\
        cameraMatrix=camera_matrix, distCoeffs=dist_coefs, flags=0)
    
    return translation_vector, rotation_vector, camera_matrix, dist_coefs


def ProjectObjectToImage(object_points, projection_matrix):
    image_points = np.transpose(projection_matrix @ np.transpose(np.c_[object_points, np.ones(len(object_points))]))
    image_points = image_points / image_points[:, 2][:, None]
    return (image_points[0:, 0:2])



def createHeatMap(lnd, radius, input_size=(512, 512), output_size=(512, 512), show=False, num_point=6000):

    hm = np.zeros(output_size, np.float)
    lnd = np.asarray(lnd)
    lnd *= np.asarray(output_size) / np.asarray(input_size)
    if any(lnd>=output_size) or any(lnd<0):
        return hm
    else:
        points = np.random.multivariate_normal(lnd, [[radius, 0],[0, radius]], num_point)
        for p in points:
            if any(p>=output_size) or any(p<0):
                continue
            hm[int(p[1]), int(p[0])] += 1
        
        hm/=np.max(hm)
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
    scaledGaussian = lambda x : np.exp(-(1/(2*sigma))*(x**2))

    if any(pt>=output_size) or any(pt<0):
        return hm
    else:
        th = 10
        for i in range(int(pt[0]-sigma/2-th), int(pt[0]+sigma/2+th)):
            for j in range(int(pt[1]-sigma/2-th), int(pt[1]+sigma/2+th)):
                if i<0 or j<0 or i>=output_size[0] or j>=output_size[1]:
                    continue

                distance = np.linalg.norm(np.array([i-pt[0],j-pt[1]]))
                scaledGaussianProb = scaledGaussian(distance)
                hm[j, i] = scaledGaussianProb
                hm[j, i] = np.clip(scaledGaussianProb*255,0,255)

    hm_blur = cv.blur(hm, (5,5))
    hm_blur = hm_blur / np.max(hm_blur)
    hm_blur[hm_blur<0.01] = 0
    return (hm_blur)


def colorHeatMap(img):
    hm_show= None
    hm_show = cv.normalize(img, hm_show, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    hm_show = cv.applyColorMap(hm_show, cv.COLORMAP_JET)
    return hm_show

def rescale_to_eight_bit(img):
    max_inten = np.max(img)
    min_inten = np.min(img)

    outimg = (img - min_inten) / (max_inten-min_inten)

    return (outimg*255).astype(int)