from itertools import cycle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Flatten, AveragePooling2D, Dense, Conv2DTranspose, ReLU, add
from tensorflow.keras import initializers
from tensorflow import keras
from  matplotlib import pyplot as plt
import matplotlib.image as mpimg
import cv2

def unpool(value, name='unpool'):
    """N-dimensional version of the unpooling operation from
    https://www.robots.ox.ac.uk/~vgg/rg/papers/Dosovitskiy_Learning_to_Generate_2015_CVPR_paper.pdf

    :param value: A Tensor of shape [b, d0, d1, ..., dn, ch]
    :return: A Tensor of shape [b, 2*d0, 2*d1, ..., 2*dn, ch]
    """
    with tf.name_scope(name) as scope:
        sh = value.get_shape().as_list()
        dim = len(sh[1:-1])
        out = (tf.reshape(value, [-1] + sh[-dim:]))
        for i in range(dim, 0, -1):
            out = tf.concat([out, tf.zeros_like(out)], i)
        out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
        out = tf.reshape(out, out_size, name=scope)
    return out

def up_projection(input_image, deconv_filters, conv5_branch1_filters, conv3_filters, conv5_branch2_filters):
  unpooling = tf.keras.layers.Lambda(unpool)(input_image)
  conv5_branch1 = Conv2D(conv5_branch1_filters, 5, padding="same", activation = "relu")(unpooling)
  conv3_branch1 = Conv2D(conv3_filters, 3, padding="same")(conv5_branch1)
  conv5_branch2 = Conv2D(conv5_branch2_filters, 3, padding="same")(unpooling)
  concat = add([conv3_branch1, conv5_branch2])
  out = tf.keras.activations.relu(concat)
  return out

def up_projection_block(skip_input, previous_output, filter):
  unpooling = up_projection(previous_output, filter, filter, filter, filter)
  # unpooling = Conv2DTranspose(1024, (2,2), strides=1, padding = "same")(previous_output)
  residual = Conv2D(filter, 2 , strides = 1, padding = "same", activation = "relu")(skip_input)
  added = add([unpooling, residual])
  return added


def CustomLoss2(y_true, y_pred):
    loss = 0
    # ISSUE: Can this two variables be initialized with in the method?
    a = tf.Variable(10., dtype=np.float32)
    c = tf.Variable(0.2, dtype=np.float32) 
    mse = tf.keras.losses.MeanSquaredError()
    loss = tf.divide(mse(y_true, y_pred),(1 + tf.math.exp(a*(c-tf.sqrt(mse(y_true, y_pred))))))
    return loss

def show_img(img, coor_pre, coor_label):
  cycol = cycle('bgrcmk')
  img = cv2.resize(img, (512, 512))
  plt.imshow(img)
  for i in range(6):
    color = next(cycol)
    print(color)
    plt.scatter(coor_pre[i, 0], coor_pre[i, 1], marker="x", color=color, s=15)
    plt.scatter(coor_label[i, 0], coor_label[i, 1], marker="o", color=color, s=15)
  plt.show()

def prediction_visualization(slice_no, l_predictions, ys, Xs):
  pre = tf.reshape(l_predictions[slice_no], [6,2])
  pts_pre = np.array(pre)

  y = tf.reshape(ys[slice_no], [6,2])
  pts_label = np.array(y)

  show_img(Xs[slice_no], pre, pts_label)