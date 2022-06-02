from itk.support.extras import origin, output, spacing
from itk.support.types import Offset
import numpy as np
import itk 
import SimpleITK as sitk
from numpy.core.fromnumeric import transpose



def GetProjectionMatrix(focal_length, trn_matrix, pp, focal_point):
    K = np.eye(3)
    K[0,0] = focal_length
    K[1,1] = focal_length
    K[0,2] = pp[0]
    K[1,2] = pp[1]

    R = np.transpose(trn_matrix[0:3, 0:3])
    F = focal_point
    Xo = trn_matrix[0:3, 3] + np.transpose(R) @ F

    P = np.zeros((3,4))
    P[0:3,0:3] = K @ R
    P[0:3, 3] = K @ R @ Xo

    P = -P / P[2,3]
    P[:, 3] *= -1

    return -P/P[2,3] 


def rotAffine(rx, ry, rz, cx, cy, cz):
    rotX = np.array([[1, 0, 0], [0, np.cos(rx), -1*np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    rotY = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-1*np.sin(ry), 0, np.cos(ry)]])
    rotZ = np.array([[np.cos(rz), -1*np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])

    rot = rotZ @ rotY @ rotX
    rotMat = np.eye(4)
    rotMat[0:3, 0:3] = rot


    trnMat = np.eye(4)
    trnMat[0, 3] = cx
    trnMat[1, 3] = cy
    trnMat[2, 3] = cz

    min_trnMat = np.eye(4)
    min_trnMat[0, 3] = -1*cx
    min_trnMat[1, 3] = -1*cy
    min_trnMat[2, 3] = -1*cz

    return(trnMat @ rotMat @ min_trnMat)



def CreateDRR(ct, focalLength=1000, outputSpacing=[1, 1, 1], outputSize=[512,512,1], outputOffset=[256,256], threshold=-2000, rotation=[0, 0, 0], translation=[0, 0, 0], write=False):
    PixelType = itk.D


    # setting up the transform
    inputSpacing = np.asarray(ct.GetSpacing())
    inputSize = np.asarray(ct.GetLargestPossibleRegion().GetSize())
    inputOrigin = np.asarray(ct.GetOrigin())

    # defining the centre of rotation as the centroid of the CT
    centroid = inputOrigin + inputSpacing * inputSize * 0.5
    # defining the focal point 
    focalPoint = np.array([centroid[0], centroid[1], centroid[2] - (focalLength*0.5)])

    # defining the principal point location 
    ox = -1 * (outputOffset[0] - (outputSize[0]*outputSpacing[0]) * 0.5)
    oy = -1 * (outputOffset[1] - (outputSize[1]*outputSpacing[1]) * 0.5)

    # defining the origin of the DRR
    outputOrigin = np.array([0, 0, 0])
    outputOrigin[0] = centroid[0] + ox - (outputSpacing[0] * (outputSize[0]-1) * 0.5)
    outputOrigin[1] = centroid[1] + oy - (outputSpacing[1] * (outputSize[1]-1) * 0.5)
    outputOrigin[2] = centroid[2] + focalLength * 0.5


    # setting up the affine roation matrix
    trnMat = np.eye(4)
    trnMat[0:3, 3] = translation[0:3]
    rotMat = rotAffine(rotation[0], rotation[1], rotation[2], centroid[0], centroid[1], centroid[2])


    finalMat = rotMat @ trnMat

    TransformType = itk.AffineTransform[itk.D, 3]
    affineTransform = TransformType.New()
    transformParams = affineTransform.GetParameters()

    for i in range(0, 3):
        for j in range(0, 3):
            transformParams[i*3 + j] = finalMat[i, j]

    for i in range(0, 3):
        transformParams[i+3 * 3] = finalMat[i, 3] 
    
    affineTransform.SetParameters(transformParams)


    # Setting up the interpolator
    interpolator = itk.RayCastInterpolateImageFunction[itk.Image[PixelType,3], PixelType].New()
    interpolator.SetFocalPoint(focalPoint)
    interpolator.SetTransform(affineTransform)
    interpolator.SetThreshold(threshold)


    # setting up the resampler 
    resampler = itk.ResampleImageFilter[itk.Image[PixelType, 3], itk.Image[PixelType, 3]].New()
    resampler.SetInput(ct)
    resampler.SetTransform(affineTransform)
    resampler.SetDefaultPixelValue(0)
    resampler.SetOutputDirection(ct.GetDirection())
    resampler.SetInterpolator(interpolator)
    resampler.SetOutputSpacing(outputSpacing)
    resampler.SetSize(outputSize)
    resampler.SetOutputOrigin(outputOrigin.astype(float))


    # setting up the rescaler 
    intensityRescaler = itk.RescaleIntensityImageFilter[itk.Image[PixelType, 3], itk.Image[PixelType, 3]].New()
    intensityRescaler.SetInput(resampler.GetOutput())
    intensityRescaler.SetOutputMinimum(0) # could bve defined by the user or maybe not!
    intensityRescaler.SetOutputMaximum(255)
    intensityRescaler.Update()

    invertFilter = itk.InvertIntensityImageFilter[itk.Image[PixelType, 3], itk.Image[PixelType, 3]].New()
    invertFilter.SetInput(intensityRescaler.GetOutput())
    invertFilter.SetMaximum(255)
    invertFilter.Update()


    drr = invertFilter.GetOutput()
    intensityRescaler.Update()

    P = GetProjectionMatrix(focal_length=focalLength, trn_matrix=finalMat, pp=outputOffset, focal_point=focalPoint)

    camera_matrix = np.eye(3)
    camera_matrix[0:0] = focalLength
    camera_matrix[1:1] = focalLength
    camera_matrix[0,2] = outputOffset[0]
    camera_matrix[1,2] = outputOffset[1]
    
    return drr, P, camera_matrix