'''
I referenced https://github.com/scheckmedia/keras-shufflenet
'''


import tensorflow as tf

def group_conv(x, in_channels, out_channels, groups, kernel_size):

    # input_tensor is channel last format
    # channels per group
    channels = in_channels // groups
    group_conv_list = []

    for i in range(groups):
        offset = i * channels
        group = tf.keras.layers.Lambda(lambda z: z[:, :, :, offset: offset + channels])(x)
        group_conv_list.append(tf.keras.layers.Conv2D(filters=int(out_channels/groups),
                                                      kernel_size=kernel_size,
                                                      strides=(1, 1))(group))
    
    output = tf.keras.layers.Concatenate()(group_conv_list)    
    
    return output


def channel_shuffle(x, groups):
    height = x.shape[1]
    width = x.shape[2]
    channels = x.shape[3]

    channels_per_group = channels // groups

    x = tf.reshape(x, [-1, height, width, groups, channels_per_group])        
    x = tf.keras.backend.permute_dimensions(x, (0, 1, 2, 4, 3))
    x = tf.reshape(x, [-1, height, width, channels])

    return x
