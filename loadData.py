import tensorflow as tf
import numpy as np
import keras

def load_data():
    img_height = 100
    img_width = 100
    batch_size = 1024

    train_dataset = keras.preprocessing.image_dataset_from_directory(
        'dataset/Training',
        labels='inferred',
        label_mode='int',
        color_mode='grayscale',
        batch_size=batch_size,
        image_size=(img_height, img_width),
        shuffle=True,
    )

    print(train_dataset.class_names)

    test_dataset = keras.preprocessing.image_dataset_from_directory(
        'dataset/Validation',
        labels='inferred',
        label_mode='int',
        color_mode='grayscale',
        batch_size=batch_size,
        image_size=(img_height, img_width),
        shuffle=False,
    )

    print("Test class names:", test_dataset.class_names)

    X_train = []
    Y_train = []
    for batch_images, batch_labels in train_dataset:
        batch_images_flat = tf.reshape(batch_images, (batch_images.shape[0], -1))
        X_train.append(batch_images_flat.numpy())
        Y_train.append(batch_labels.numpy())
        
    X_train = np.vstack(X_train)
    Y_train = np.hstack(Y_train) 

    X_test = []
    Y_test = []

    for batch_images, batch_labels in test_dataset:
        batch_images_flat = tf.reshape(batch_images, (batch_images.shape[0], -1))
        X_test.append(batch_images_flat.numpy())
        Y_test.append(batch_labels.numpy())
        
    X_test = np.vstack(X_test)
    Y_test = np.hstack(Y_test)
    
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    return X_train, Y_train, X_test, Y_test