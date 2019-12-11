import tensorflow as tf



class VGG16(tf.keras.Model):
    def __init__(self, nb_classes):
        super(VGG16, self).__init__()

        self.nb_class = nb_classes
        
        self.conv1_1 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')
        self.conv1_2 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')
        self.max_pool1 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))
        
        self.conv2_1 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')
        self.conv2_2 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')
        self.max_pool2 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))

        self.conv3_1 = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')
        self.conv3_2 = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')
        self.conv3_3 = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')
        self.max_pool3 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))

        self.conv4_1 = tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu')
        self.conv4_2 = tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu')
        self.conv4_3 = tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu')
        self.max_pool4 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))

        self.conv5_1 = tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu')
        self.conv5_2 = tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu')
        self.conv5_3 = tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu')
        self.max_pool5 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))

        self.flat = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(4096, activation='relu')
        self.dense2 = tf.keras.layers.Dense(4096, activation='relu')
        self.dense3 = tf.keras.layers.Dense(nb_classes, activation='relu')

    def call(self, x, training=False):
        
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.max_pool1(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.max_pool2(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.max_pool3(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.max_pool4(x)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.max_pool5(x)

        x = self.flat(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)

        return x
    


model = VGG16(1000)
model.build((1, 224, 224, 3))
model.summary()
