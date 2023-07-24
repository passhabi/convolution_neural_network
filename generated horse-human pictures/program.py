# download the zip files:
# https://storage.googleapis.com/tensorflow-1-public/course2/week3/horse-or-human.zip
# https://storage.googleapis.com/tensorflow-1-public/course2/week3/validation-horse-or-human.zip
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from zipfile import ZipFile
import os

# extracting the data
with ZipFile('horse-or-human.zip') as zipfile:
    zipfile.extractall('train_data')

with ZipFile('validation-horse-or-human.zip') as zipfile:
    zipfile.extractall('valid_data')


train_dataset = tf.keras.utils.image_dataset_from_directory(
    "train_data",
    image_size=(300, 300),
    batch_size=128,
    label_mode='binary'
)

valid_dataset = tf.keras.utils.image_dataset_from_directory(
    "valid_data",
    image_size=(300, 300),
    batch_size=128,
    label_mode='binary'
)

# datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
# train_data_gen = datagen.flow_from_dataframe(
#     'train-data',
#     target_size=(300, 300, 1),
#     batch_size=128,
#     class_mode='binary'
# )


# creating the model:
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(scale=1/255),
    tf.keras.layers.Conv2D(filters=8,
                           kernel_size=(3, 3),
                           input_shape=(300, 300, 3),
                           activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),

    tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),

    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),

    tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),

    tf.keras.layers.Flatten(),
    # tf.keras.layers.Rescaling(scale=1 / 255),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='sigmoid'),
])

model.build((None, 300, 300, 3))

model.summary()
model.compile(loss=tf.keras.losses.binary_crossentropy,
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'], )

model.fit(train_dataset, epochs=12)  # %100 accuracy on train data


# todo: check for overfiting..
# prediction:
# os.mkdir("test")  # put test files in this folder.
load_img = tf.keras.utils.load_img
img_to_array = tf.keras.utils.img_to_array

for img_path in os.listdir('test'):

    img = load_img(os.path.join('test', img_path), target_size=(300, 300, 3), keep_aspect_ratio=True)
    x = img_to_array(img)
    # x = x / 255  # normalize the data # have done that in model!
    x = np.expand_dims(x, axis=0)
    x = np.vstack([x])

    prediction = model.predict(x)
    if prediction[0] > 0.5:
        print(prediction, "The file\33[34m", img_path, "\33[38mis a human")
    else:
        print(prediction, "The file\33[34m", img_path, "\33[38mis a horse")
