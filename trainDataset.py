import numpy as np
import tensorflow as tf
import keras
from keras import Sequential
from keras import layers
from keras import activations 
import loadData


x_train, y_train, x_test, y_test = loadData.load_data()
model = Sequential(
    [
        layers.Dense(64, activation=activations.relu, name = "l1"),
        layers.Dense(32, activation = activations.relu, name = "l2"),
        layers.Dense(16, activation = activations.relu, name = "l3"),
        layers.Dense(1, activation = activations.linear, name = "l4"),
    ], name="biometric_attribute_classification"
)

model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=True), optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy']),

history = model.fit(
    x_train, y_train,
    epochs=100
)

loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")
model.save("biometric_attribute_classification.keras") 
