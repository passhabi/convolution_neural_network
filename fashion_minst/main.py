
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_x, train_y), (dev_x, dev_y) = fashion_mnist.load_data()

labels = ['T-Shirt/Top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

print("# training examples (m)", train_x.shape[0])
print("Height × Width:", train_x.shape[1:])

# looking at the data:
idx = 0
plt.imshow(train_x[idx])
plt.title(f"This is a {labels[train_y[idx]]} {[train_y[idx]]}")
plt.show()


