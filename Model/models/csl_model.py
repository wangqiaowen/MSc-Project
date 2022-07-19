from Model.models.l_model import LModel
from Model.Helpers.models.utils import *
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, concatenate, UpSampling2D, BatchNormalization, Flatten, AveragePooling2D, Dense, Conv2DTranspose, ReLU, add
# from tensorflow_addons.layers import MaxUnpooling2D
from sklearn.model_selection import train_test_split
from tensorflow.keras import initializers
from tensorflow import keras

class CSLModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
    
    @staticmethod
    def csl_model(encoder_model, encoder_1024, encoder_512, encoder_256, no_landmarks):
        fconv_1 = Conv2D(1024, 1 , padding = "same", activation = "relu")(encoder_model.layers[-1].output)
        # fconv_1 = residual_block(encoder_model.layers[-1].output, 1024)

        o1 = up_projection_block(encoder_1024, fconv_1, 512)

        # unpooling_1 = up_projection(fconv_1, 512, 512, 512, 512)
        # residual_1 = Conv2D(512, 1 , strides = 1, padding = "same", activation = "relu")(encoder_1024)
        # concat_1 = add([unpooling_1, residual_1])

        o2 = up_projection_block(encoder_512, o1, 256)

        # unpooling_2 = up_projection(concat_1, 256, 256, 256, 256)
        # residual_2 = Conv2D(256, 1 , strides = 1, padding = "same", activation = "relu")(encoder_512)
        # concat_2 = add([unpooling_2, residual_2])

        o3 = up_projection_block(encoder_256, o2, 128)

        # unpooling_3 = up_projection(concat_2, 128, 128, 128, 128)
        # residual_3 = Conv2D(128, 1 , strides = 1, padding = "same", activation = "relu")(encoder_256)
        # concat_3 = add([unpooling_3, residual_3])

        unpooling_3 = up_projection(o3, 64, 64, 64, 64)

        # unpooling_3 = up_projection(concat_3, 64, 64, 64, 64)


        seg_1 = Conv2D(2, 1, padding="same")(unpooling_3)  # 2: number of classes in segmentation task: one for tool, one for background
        
        fconv_2 = Conv2D(32, 2, padding="same", activation = "relu")(unpooling_3)

        concat = concatenate([seg_1, fconv_2])

        seg_output = tf.keras.layers.Activation(activation='softmax',  name="segmentation_branch")(seg_1)

        output = Conv2D(no_landmarks, 1, padding="same", activation="linear", name="cslmodel_output")(concat)
        # output = Conv2D(n, 2, padding="same", activation="relu", name="cslmodel_output")(fconv_2)

        cslmodel = Model(inputs=encoder_model.inputs, outputs=[output, seg_output], name="cslmodel")
        # return output
        return cslmodel