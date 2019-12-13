import tensorflow as tf


class SepConv(tf.keras.Model):
    def __init__(self, in_channels, out_channels):
        super(SepConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.dw_conv1 = tf.keras.layers.DepthwiseConv2D((3, 3), 
                                                        strides=(1, 1), 
                                                        padding='same',
                                                        use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()

        self.conv2 = tf.keras.layers.Conv2D(self.out_channels, (1, 1), padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()        

    def call(self, input_tensor, training=False):
        x = self.dw_conv1(input_tensor)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        return x


class MBConv_SE(tf.keras.Model):
    def __init__(self, expansion, in_channels, out_channels, kernel_size, strides):
        super(MBConv_SE, self).__init__()
        
        self.expansion = expansion
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.strides = strides


        self.conv1 = tf.keras.layers.Conv2D(self.in_channels*self.expansion, 
                                            kernel_size=(1, 1),
                                            padding='same',
                                            use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()

        
        self.dw_conv2 = tf.keras.layers.DepthwiseConv2D(kernel_size=self.kernel_size,
                                                        strides=self.strides,
                                                        padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.ReLU()


        self.se_gap3 = tf.keras.layers.GlobalAveragePooling2D()
        self.se_fc3_1 = tf.keras.layers.Dense((self.in_channels*self.expansion)//4)
        self.se_relu3 = tf.keras.layers.ReLU()
        self.se_fc3_2 = tf.keras.layers.Dense((self.in_channels*self.expansion), activation='sigmoid')
        self.multiply3 = tf.keras.layers.Multiply()


        self.conv4 = tf.keras.layers.Conv2D(self.out_channels,
                                            kernel_size=(1, 1),
                                            padding='same')
        self.bn4 = tf.keras.layers.BatchNormalization()

        if self.in_channels == self.out_channels:
            self.add5 = tf.keras.layers.Add()


    def call(self, input_tensor, training=False):
        x = self.conv1(input_tensor)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.dw_conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x_se_input = self.se_gap3(x)
        x_se = self.se_fc3_1(x_se_input)
        x_se = self.se_relu3(x_se)
        x_se = self.se_fc3_2(x_se)
        x_se = self.multiply3([x_se, x])

        x = self.conv4(x_se)
        x = self.bn4(x)
        if self.in_channels == self.out_channels:
            x = self.add5([input_tensor, x])

        return x




class MBConv(tf.keras.Model):
    def __init__(self, expansion, in_channels, out_channels, kernel_size, strides):
        super(MBConv, self).__init__()

        self.expansion = expansion
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.strides = strides

        self.conv1 = tf.keras.layers.Conv2D((self.in_channels*self.expansion), 
                                            kernel_size=(1, 1), 
                                            padding='same',
                                            use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()        
        self.relu1 = tf.keras.layers.ReLU()



        self.dw_conv2 = tf.keras.layers.DepthwiseConv2D(kernel_size=self.kernel_size, 
                                                        strides=self.strides, 
                                                        padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.ReLU()



        self.conv3 = tf.keras.layers.Conv2D(self.out_channels, kernel_size=(1, 1), padding='same')
        self.bn3 = tf.keras.layers.BatchNormalization()
        
        if self.in_channels == self.out_channels:
            self.add4 = tf.keras.layers.Add()
    

    def call(self, input_tensor, training=False):
        x = self.conv1(input_tensor)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.dw_conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        if self.in_channels == self.out_channels:
            x = self.add4([input_tensor, x])

        return x






class MnasNet(tf.keras.Model):
    def __init__(self, nb_classes):
        super(MnasNet, self).__init__()

        self.nb_classes = nb_classes

        # First conv layer
        self.conv_1 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='same')


        # SepConv x1
        self.SepConv_2 = tf.keras.Sequential()
        self.SepConv_2.add(SepConv(in_channels=32, out_channels=16))

        # MBConv6 (k3x3) x2
        self.MBConv_3 = tf.keras.Sequential()
        self.MBConv_3.add(MBConv(expansion=6, in_channels=16, out_channels=24, kernel_size=(3, 3), strides=(2, 2)))
        self.MBConv_3.add(MBConv(expansion=6, in_channels=24, out_channels=24, kernel_size=(3, 3), strides=(1, 1)))

        # MBConv3_SE (k5x5) x3
        self.MBConv_SE_4 = tf.keras.Sequential()
        self.MBConv_SE_4.add(MBConv_SE(expansion=3, in_channels=24, out_channels=40, kernel_size=(5, 5), strides=(2, 2)))
        self.MBConv_SE_4.add(MBConv_SE(expansion=3, in_channels=40, out_channels=40, kernel_size=(5, 5), strides=(1, 1)))
        self.MBConv_SE_4.add(MBConv_SE(expansion=3, in_channels=40, out_channels=40, kernel_size=(5, 5), strides=(1, 1)))

        # MBConv6 (k3x3) x4
        self.MBConv_5 = tf.keras.Sequential()
        self.MBConv_5.add(MBConv(expansion=6, in_channels=40, out_channels=80, kernel_size=(3, 3), strides=(2, 2)))
        self.MBConv_5.add(MBConv(expansion=6, in_channels=80, out_channels=80, kernel_size=(3, 3), strides=(1, 1)))
        self.MBConv_5.add(MBConv(expansion=6, in_channels=80, out_channels=80, kernel_size=(3, 3), strides=(1, 1)))
        self.MBConv_5.add(MBConv(expansion=6, in_channels=80, out_channels=80, kernel_size=(3, 3), strides=(1, 1)))

        # MBConv6_SE (k3x3) x2
        self.MBConv_SE_6 = tf.keras.Sequential()
        self.MBConv_SE_6.add(MBConv_SE(expansion=6, in_channels=80, out_channels=112, kernel_size=(3, 3), strides=(1, 1)))
        self.MBConv_SE_6.add(MBConv_SE(expansion=6, in_channels=112, out_channels=112, kernel_size=(3, 3), strides=(1, 1)))

        # MBConv6_SE (k5x5) x3
        self.MBConv_SE_7 = tf.keras.Sequential()
        self.MBConv_SE_7.add(MBConv_SE(expansion=6, in_channels=112, out_channels=160, kernel_size=(5, 5), strides=(2, 2)))
        self.MBConv_SE_7.add(MBConv_SE(expansion=6, in_channels=160, out_channels=160, kernel_size=(5, 5), strides=(1, 1)))
        self.MBConv_SE_7.add(MBConv_SE(expansion=6, in_channels=160, out_channels=160, kernel_size=(5, 5), strides=(1, 1)))

        # MBConv6 (k3x3) x1
        self.MBConv_8 = tf.keras.Sequential()
        self.MBConv_8.add(MBConv(expansion=6, in_channels=160, out_channels=320, kernel_size=(3, 3), strides=(1, 1)))

        # Pooling, FC
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(self.nb_classes, activation='softmax')


    def call(self, input_tensor, training=False):
        x = self.conv_1(input_tensor)
        x = self.SepConv_2(x)
        x = self.MBConv_3(x)
        x = self.MBConv_SE_4(x)
        x = self.MBConv_5(x)
        x = self.MBConv_SE_6(x)
        x = self.MBConv_SE_7(x)
        x = self.MBConv_8(x)
        x = self.pool(x)
        x = self.fc(x)

        return x


model = MnasNet(1000)
model.build((1, 224, 224, 3))
model.summary()
