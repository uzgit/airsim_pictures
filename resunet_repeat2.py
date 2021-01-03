import tensorflow
from keras import backend as K
from keras import layers
from keras.layers import Add, Input, UpSampling2D, Concatenate
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from keras.callbacks import Callback, ModelCheckpoint
import matplotlib.pyplot as plt
import keras

k_size = 3

def tversky(y_true, y_pred, smooth=1e-6):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = tensorflow.reduce_sum(y_true_pos * y_pred_pos)
    false_neg = tensorflow.reduce_sum(y_true_pos * (1-y_pred_pos))
    false_pos = tensorflow.reduce_sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)

def focal_tversky_loss(y_true,y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return tensorflow.keras.backend.pow((1-pt_1), gamma)

def bn_act(x, act=True):
    'batch normalization layer with an optinal activation layer'
    # print("in bn_act")
    # x = tf.keras.layers.BatchNormalization() (x)
    x = keras.layers.BatchNormalization()(x)
    # print("in bn_act 2")
    if act == True:
        x = keras.layers.Activation('relu')(x)
    return x


def conv_block(x, filters, kernel_size=3, padding='same', strides=1):
    'convolutional layer which always uses the batch normalization layer'
    conv = bn_act(x)
    # print("here1")
    conv = Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)
    return conv


def stem(x, filters, kernel_size=3, padding='same', strides=1):
    # print("one")
    conv = Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    # print("two")
    conv = conv_block(conv, filters, kernel_size, padding, strides)
    # print("three")
    shortcut = Conv2D(filters, kernel_size=1, padding=padding, strides=strides)(x)
    # print("four")
    shortcut = bn_act(shortcut, act=False)
    # print("five")
    output = Add()([conv, shortcut])
    # print("six")
    return output


def residual_block(x, filters, kernel_size=3, padding='same', strides=1):
    res = conv_block(x, filters, k_size, padding, strides)
    res = conv_block(res, filters, k_size, padding, 1)
    shortcut = Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)
    output = Add()([shortcut, res])
    return output


def upsample_concat_block(x, xskip):
    u = UpSampling2D((2, 2))(x)
    c = Concatenate()([u, xskip])
    return c


def ResUNet(img_h, img_w):
    # print(img_h)

    f = [16, 32, 64, 128, 256]
    inputs = Input((img_h, img_w, 1))
    # print(inputs)

    ## Encoder
    e0 = inputs
    # print(e0)
    # print("right before first stem")
    e1 = stem(e0, f[0])
    # print("right after first stem")
    e2 = residual_block(e1, f[1], strides=2)
    e3 = residual_block(e2, f[2], strides=2)
    e4 = residual_block(e3, f[3], strides=2)
    e5 = residual_block(e4, f[4], strides=2)

    ## Bridge
    b0 = conv_block(e5, f[4], strides=1)
    b1 = conv_block(b0, f[4], strides=1)

    ## Decoder
    u1 = upsample_concat_block(b1, e4)
    d1 = residual_block(u1, f[4])

    u2 = upsample_concat_block(d1, e3)
    d2 = residual_block(u2, f[3])

    u3 = upsample_concat_block(d2, e2)
    d3 = residual_block(u3, f[2])

    u4 = upsample_concat_block(d3, e1)
    d4 = residual_block(u4, f[1])

    outputs = keras.layers.Conv2D(4, (1, 1), padding="same", activation="sigmoid")(d4)
    model = keras.models.Model(inputs, outputs)
    return model


# model = build_model((256,1600,1))
model = ResUNet(img_h=256, img_w=800)
adam = keras.optimizers.Adam(lr=0.05, epsilon=0.1)
model.compile(optimizer=adam, loss=focal_tversky_loss, metrics=[tversky])
# model.compile(optimizer=adam, loss=focal_tversky_loss, metrics=[tversky])

model.summary()