
import tensorflow as tf
import matplotlib.pyplot as plt

fashion_mnist = tf.keras.datasets.fashion_mnist
(x, y), (x_test, y_test) = fashion_mnist.load_data()  # x, y variables corresponds to training data by default.

labels = ['T-Shirt/Top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

print("# training examples (m)", x.shape[0])
print("Height × Width:", x.shape[1:])

# looking at the data:
idx = 4
plt.imshow(x[idx])
plt.title(f"This is a {labels[y[idx]]} {[y[idx]]}")
plt.show()

# CNN
# normalizing data:
x = x / 255
x_test = x_test / 255

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=16,
                           kernel_size=(3, 3),
                           padding="same",
                           input_shape=x.shape[1:] + (1,),
                           activation='relu'),
    tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding="same", activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation='relu'),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    # tf.keras.layers.Rescaling(scale=1 / 255),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=10, activation='softmax'),
])
model.summary()
model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.0055, ),
              metrics=['accuracy'], )

model.fit(x, y, batch_size=1024,  epochs=20)  # loss: 0.0680 - accuracy: 0.9745
