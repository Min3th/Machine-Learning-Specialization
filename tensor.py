import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense,Input
from tensorflow.keras import Sequential
from tensorflow.keras.losses import MeanSquaredError,BinaryCrossentropy
from tensorflow.keras.activations import sigmoid

X_train = np.array([[1.0],[2.0]], dtype=np.float32)
Y_train = np.array([[300.0],[500.0]], dtype=np.float32)

fig, ax = plt.subplots(1,1)
ax.scatter(X_train,Y_train,marker = 'x', c='r', label = "Data Points")
ax.legend(fontsize='xx-large')
ax.set_ylabel('Price (in 1000s of dollars)',fontsize = 'xx-large')
ax.set_xlabel('Size (1000 sqft)', fontsize='xx-large')
#plt.show()

linear_layer = tf.keras.layers.Dense(units=1,activation = 'linear',)

linear_layer.get_weights()

a1 = linear_layer(X_train[0].reshape(1,1))
print(a1)

w,b = linear_layer.get_weights()
print(f"w={w},b={b}")