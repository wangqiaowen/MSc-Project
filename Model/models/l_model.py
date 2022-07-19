import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Flatten, AveragePooling2D, Dense, Conv2DTranspose, ReLU, add
from tensorflow.keras import initializers
from tensorflow import keras
import argparse
import pickle

class LModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
    
    @staticmethod
    def residual_block(resnet_output, filters):
        conv_r = Conv2D(filters, 1 , strides = 2, padding = "same", activation = "relu", kernel_initializer=initializers.RandomNormal(stddev=0.01))(resnet_output)
        conv1 = Conv2D(filters, 3, strides = 2, padding = "same", activation = "relu", kernel_initializer=initializers.RandomNormal(stddev=0.01))(resnet_output)
        bn1 = BatchNormalization()(conv1)
        conv2 = Conv2D(filters, 3,  padding = "same", activation = "relu",kernel_initializer=initializers.RandomNormal(stddev=0.01))(bn1)
        bn2 = BatchNormalization()(conv2)
        concat = add([conv_r, bn2])
        return concat

    @staticmethod
    def l_model(resnet_output, filters, no_landmarks):
      concat = LModel.residual_block(resnet_output, filters)
      p = AveragePooling2D((4, 4))(concat) # Remember to modify the pooling stride
      flat1 = Flatten()(p)
      output = Dense(2*no_landmarks, name="localization_output", kernel_initializer=initializers.RandomNormal(stddev=0.01))(flat1)
      return output

    @staticmethod
    def build(encoder_model, no_landmarks, filters ):
      model = Model(inputs=encoder_model.inputs, outputs=LModel.l_model(encoder_model.layers[-1].output,filters, no_landmarks))
      return model
 
    @staticmethod
    def main():

        ap = argparse.ArgumentParser()
        ap.add_argument("-i", "--img", required=True, help="path to img dataset (i.e., file path of images)")
        # {'img': '/Users/qiaowenwang/Downloads/IMG_4589.PNG'}
        ap.add_argument("-c", "--coor", required=True, help="path to coor dataset (i.e., file path of coors)")
        ap.add_argument("-vi", "--valimg", required=True, help="path to validation img dataset")
        ap.add_argument("-vc", "--valcoor", required=True, help="path to validation coor dataset")
        ap.add_argument("-mo", "--model_output", required=True, help="path to store output")
        ap.add_argument("-ho", "--history_output", required=True, help="path to store output")
        ap.add_argument("-n", "--coor_normalize_factor", required=True, help="size of the mask")
        ap.add_argument("-p", "--plot", type=str, default="output",help="base filename for generated plots")
        args = vars(ap.parse_args())

        img_path = args['img']
        coor_path = args['coor']
        valimg_path = args['valimg']
        valcoor_path = args['valcoor']
        model_output_path = args['model_output']
        history_output_path = args['history_output']
        coor_normalize_factor = args['coor_normalize_factor']

        img_array_np = np.load(img_path)
        coor_array_np = np.load(coor_path)/coor_normalize_factor
        val_img_array_np = np.load(valimg_path)
        val_coor_array_np = np.load(valcoor_path)/coor_normalize_factor

        batch_size = 8
        epochs = 200
        model = LModel.build(tf.keras.applications.ResNet50(input_shape=(512,512,3),include_top=False), 6, 2048)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        opt = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
        model.compile(optimizer=opt,loss='mean_squared_error', metrics=["accuracy"])
        history = model.fit(img_array_np, coor_array_np, batch_size=batch_size, epochs=epochs, shuffle=True, validation_data = (val_img_array_np, val_coor_array_np), callbacks=[early_stopping])
        model.save_weights(model_output_path)
        with open(history_output_path, 'wb') as file_pi:
            pickle.dump(history.history, file_pi)


if __name__ == "__main__":
    LModel.main()
    # batch_size = 8
    # epochs = 200
    # model = LModel.build(tf.keras.applications.ResNet50(input_shape=(512,512,3),include_top=False), 6, 2048)
    # early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    # opt = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
    # model.compile(optimizer=opt,loss='mean_squared_error', metrics=["accuracy"])
    # history = model.fit(img_array_np, coor_array_np, batch_size=batch_size, epochs=epochs, shuffle=True, validation_data = (val_img_array_np, val_coor_array_np), callbacks=[early_stopping])