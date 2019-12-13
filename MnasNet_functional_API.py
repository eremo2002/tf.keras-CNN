import keras
from keras.layers import *
from keras import Model


def SepConv(inputs, in_channels, out_channels):
    x = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters=out_channels, kernel_size=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)

    return x


def MBConv_SE(inputs, expansion, in_channels, out_channels, kernel_size, strides):
    x = Conv2D(in_channels*expansion, 
                kernel_size=(1, 1), 
                padding='same', 
                use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)


    x = DepthwiseConv2D(kernel_size=kernel_size, 
                        strides=strides, 
                        padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    se = GlobalAveragePooling2D()(x)
    se = Reshape((1, 1, in_channels*expansion))(se)
    se = Dense((in_channels*expansion)//4, activation='relu')(se)
    se = Dense((in_channels*expansion), activation='sigmoid')(se)
    se = Multiply()([x, se])

    x = Conv2D(out_channels, kernel_size=(1, 1), padding='same')(se)
    x = BatchNormalization()(x)

    if in_channels == out_channels:
        x = Add()([x, inputs])
    
    return x


def MBConv(inputs, expansion, in_channels, out_channels, kernel_size, strides):
    x = Conv2D(in_channels*expansion, 
                kernel_size=(1, 1), 
                padding='same', 
                use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)


    x = DepthwiseConv2D(kernel_size=kernel_size,
                        strides=strides,
                        padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    
    x = Conv2D(out_channels, 
                kernel_size=(1, 1), 
                padding='same')(x)
    x = BatchNormalization()(x)


    if in_channels == out_channels:
        x = Add()([x, inputs])

    return x




input_tensor = Input(shape=(224, 224, 3), dtype='float32', name='input')
x = Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='same')(input_tensor)

# SepConv x1
x = SepConv(inputs=x, in_channels=32, out_channels=16)

# MBConv6 (k3x3) x2
x = MBConv(inputs=x, expansion=6, in_channels=16, out_channels=24, kernel_size=(3, 3), strides=(2, 2))
x = MBConv(inputs=x, expansion=6, in_channels=16, out_channels=24, kernel_size=(3, 3), strides=(1, 1))

# MBConv3_SE (k5x5) x3
x = MBConv_SE(inputs=x, expansion=3, in_channels=24, out_channels=40, kernel_size=(5, 5), strides=(2, 2))
x = MBConv_SE(inputs=x, expansion=3, in_channels=40, out_channels=40, kernel_size=(5, 5), strides=(1, 1))
x = MBConv_SE(inputs=x, expansion=3, in_channels=40, out_channels=40, kernel_size=(5, 5), strides=(1, 1))

# MBConv6 (k3x3) x4
x = MBConv(inputs=x, expansion=6, in_channels=40, out_channels=80, kernel_size=(3, 3), strides=(2, 2))
x = MBConv(inputs=x, expansion=6, in_channels=80, out_channels=80, kernel_size=(3, 3), strides=(1, 1))
x = MBConv(inputs=x, expansion=6, in_channels=80, out_channels=80, kernel_size=(3, 3), strides=(1, 1))
x = MBConv(inputs=x, expansion=6, in_channels=80, out_channels=80, kernel_size=(3, 3), strides=(1, 1))

# MBConv6_SE (k3x3) x2
x = MBConv_SE(inputs=x, expansion=6, in_channels=80, out_channels=112, kernel_size=(3, 3), strides=(1, 1))
x = MBConv_SE(inputs=x, expansion=6, in_channels=112, out_channels=112, kernel_size=(3, 3), strides=(1, 1))

# MBConv6_SE (k5x5) x3
x = MBConv_SE(inputs=x, expansion=6, in_channels=112, out_channels=160, kernel_size=(5, 5), strides=(2, 2))
x = MBConv_SE(inputs=x, expansion=6, in_channels=160, out_channels=160, kernel_size=(5, 5), strides=(1, 1))
x = MBConv_SE(inputs=x, expansion=6, in_channels=160, out_channels=160, kernel_size=(5, 5), strides=(1, 1))

# MBConv6 (k3x3) x1
x = MBConv(inputs=x, expansion=6, in_channels=160, out_channels=320, kernel_size=(3, 3), strides=(1, 1))

x = GlobalAveragePooling2D()(x)
output_tensor = Dense(6, activation='softmax')(x)

model = Model(input_tensor, output_tensor)
model.summary()



