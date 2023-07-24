# download the zip files:
# https://storage.googleapis.com/tensorflow-1-public/course2/week3/horse-or-human.zip
# https://storage.googleapis.com/tensorflow-1-public/course2/week3/validation-horse-or-human.zip

import tensorflow as tf
import matplotlib.pyplot as plt
from zipfile import ZipFile

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


# creating the model:
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(scale=1/255),
    tf.keras.layers.Conv2D(filters=16,
                           kernel_size=(3, 3),
                           input_shape=(300, 300, 3),
                           activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),

    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),

    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),

    tf.keras.layers.Flatten(),
    # tf.keras.layers.Rescaling(scale=1 / 255),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=10, activation='softmax'),
])
model.build(trai)
model.summary()
model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.0055, ),
              metrics=['accuracy'], )

model.fit(x, y, batch_size=1024,  epochs=20)  # loss: 0.0680 - accuracy: 0.9745
