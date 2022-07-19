from Model.models.l_model import LModel
from Model.Helpers.models.utils import up_projection
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, concatenate, UpSampling2D, BatchNormalization, Flatten, AveragePooling2D, Dense, Conv2DTranspose, ReLU, add
# from tensorflow_addons.layers import MaxUnpooling2D
from sklearn.model_selection import train_test_split
from tensorflow.keras import initializers
from tensorflow import keras


class SLModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
    
    @staticmethod
    def s_branch(resnet_output):
        d1 = up_projection(resnet_output, 512, 512, 512, 512)
        d2 = up_projection(d1, 256, 256, 256, 256)
        d3 = up_projection(d2, 128, 128, 128, 128)
        d4 = up_projection(d3, 64, 64, 64, 64)
        # outputs = Conv2D(1, 1, padding="same", activation="sigmoid", name="segmentation_output")(d4)

        # for 2 channels, categorical masks
        outputs = Conv2D(2, 1, padding="same", activation="softmax", name="segmentation_output")(d4) 

        # outputs = tf.keras.layers.Reshape((2, 256*256))(outputs)
        # outputs = tf.keras.layers.Permute((2,1), name="permute_output")(outputs)
        # outputs = Activation("softmax")(outputs)

        return outputs

    @staticmethod
    def l_branch(resnet_output, no_landmarks):
        return LModel.l_model(resnet_output,1024,no_landmarks)

    @staticmethod
    def slmodel_build(encoder_model):

        inputs = encoder_model.inputs
        resnet_output = encoder_model.layers[-1].output

        l_branch = SLModel.l_branch(resnet_output)
        s_branch = SLModel.s_branch(resnet_output)
        
        slmodel = Model(inputs=inputs, outputs=[l_branch, s_branch],name="slmodel")

        return slmodel