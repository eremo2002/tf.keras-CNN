import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers, initializers, regularizers, metrics

from mobilenetV1_subclass import *
from resnet50_subclass import *
from vgg16_subclassing import *

tf.reset_default_graph()
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
tf.keras.backend.set_session(tf.Session(config=config))


np.random.seed(5)
img_size = (128, 128)
epochs = 100
batch_size = 64

train_df = pd.read_csv("train.csv")
valid_df = pd.read_csv("val.csv")

train_df['label'] = train_df['label'].astype(str)
valid_df['label'] = valid_df['label'].astype(str)


train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip = True, 
    vertical_flip = False,
    zoom_range=0.10)

test_datagen = ImageDataGenerator(
    rescale=1./255)


train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory='data/',
        x_col="filename",
        y_col="label",
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_dataframe(
        dataframe=valid_df,
        directory='data/',
        x_col="filename",
        y_col="label",
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical')





# model = MobilenetV1(1000)
# model = VGG16(1000)
model = ResNet50(1000)

model.build((batch_size, 128, 128, 3))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-4), metrics=['acc'])


checkpoint = ModelCheckpoint(filepath='./weight/epoch-{epoch:04d}-acc_{acc:.4f}-val_loss_{val_loss:.4f}-val_acc_{val_acc:.4f}.h5', 
                            monitor='val_loss', 
                            verbose=1, 
                            save_weights_only=True)

# lr_check = LRTensorBoard(log_dir='./weight')


history = model.fit_generator(generator=train_generator,
                                steps_per_epoch=int(train_generator.n / train_generator.batch_size),
                                epochs=epochs, 
                                validation_data=validation_generator,
                                validation_steps=int(validation_generator.n / validation_generator.batch_size),
                                verbose=1, shuffle=True, callbacks=[checkpoint])
