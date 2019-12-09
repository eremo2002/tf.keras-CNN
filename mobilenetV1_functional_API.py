import keras
from keras.layers import *
from keras import Model

def depthwise_bn_relu(x, s, padd):
    x = DepthwiseConv2D((3, 3), strides=(s, s), padding=padd, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def pointwise_bn_relu(x, number_of_filter):
    x = Conv2D(number_of_filter, (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x






input_tensor = Input(shape=(224, 224, 3), dtype='float32', name='input')

x = ZeroPadding2D(padding=(2,2))(input_tensor)
x = Conv2D(32, (3, 3), strides=(2, 2), padding='valid', use_bias=False)(x)

x = BatchNormalization()(x)
x = Activation('relu')(x)


x = depthwise_bn_relu(x, 1, 'same')
x = pointwise_bn_relu(x, 64)
x = ZeroPadding2D(padding=((0, 1), (0, 1)))(x)



x = depthwise_bn_relu(x, 2, 'valid')
x = pointwise_bn_relu(x, 128)


x = depthwise_bn_relu(x, 1, 'same')
x = pointwise_bn_relu(x, 128)
x = ZeroPadding2D(padding=((0, 1), (0, 1)))(x)


x = depthwise_bn_relu(x, 2, 'valid')
x = pointwise_bn_relu(x, 256)

x = depthwise_bn_relu(x, 1, 'same')
x = pointwise_bn_relu(x, (256))
x = ZeroPadding2D(padding=((0, 1), (0, 1)))(x)


x = depthwise_bn_relu(x, 2, 'valid')
x = pointwise_bn_relu(x, 512)

for _ in range(5):
    x = depthwise_bn_relu(x, 1, 'same')
    x = pointwise_bn_relu(x, 512)

x = ZeroPadding2D(padding=((0, 1), (0, 1)))(x)

x = depthwise_bn_relu(x, 2, 'valid')
x = pointwise_bn_relu(x, 1024)

x = depthwise_bn_relu(x, 2, 'same')
x = pointwise_bn_relu(x, 1024)

x = GlobalAveragePooling2D()(x)
x = Reshape((1, 1, 1024))(x)
x = Dropout(0.001)(x)
x = Conv2D(1000, (1, 1), strides=(1, 1), padding='same')(x)
x = Activation('softmax')(x)

output_tensor = Reshape((1000,))(x)


my_mobile = Model(input_tensor, output_tensor)
my_mobile.summary()