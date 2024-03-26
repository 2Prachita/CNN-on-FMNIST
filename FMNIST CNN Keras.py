import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical

# 1. Import Dataset
# 2. Preprocess Dataset
# 3. Create training model
# 4. Train model
# 5. Test model
# 6. Evaluate Accuracy



####### 1. Importing dataset

fashion_mnist = tf.keras.datasets.fashion_mnist

#This is a dataset of 60,000 28x28 grayscale images of 10 fashion categories, along with a test set of 10,000 images. This dataset can be used as a drop-in replacement for MNIST.
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images[:,:,:,np.newaxis]
test_images = test_images[:,:,:,np.newaxis]

####### 2. Preprocess Dataset

#Scale these values to a range of 0 to 1 before feeding them to the neural network model. To do so, divide the values by 255.
train_images = train_images / 255.
test_images = test_images / 255.

train_images = train_images.astype(np.float32)
test_images = test_images.astype(np.float32)
train_labels = train_labels.astype(np.float32)
test_labels = test_labels.astype(np.float32)


####### 3. Create training model Dropout == 0.2

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=16,kernel_size=(5, 5),strides=(1, 1),padding="SAME", 
                     activation=tf.nn.relu,kernel_regularizer=tf.keras.regularizers.l2(l=1e-3)),
    tf.keras.layers.Conv2D(filters=32,kernel_size=(3, 3),strides=(1, 1),padding="SAME", 
                     activation=tf.nn.relu,kernel_regularizer=tf.keras.regularizers.l2(l=1e-3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides=1,padding="SAME"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(l=1e-3)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10,kernel_regularizer=tf.keras.regularizers.l2(l=1e-3))
]) 

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

####### 4. Train model

model.fit(train_images, train_labels, epochs=10)


####### 5. Test model

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

####### 3. Create training model1 Dropout == 0.2

model1 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=16,kernel_size=(5, 5),strides=(1, 1),padding="SAME", 
                     activation=tf.nn.relu,kernel_regularizer=tf.keras.regularizers.l2(l=1e-3)),
    tf.keras.layers.Conv2D(filters=32,kernel_size=(3, 3),strides=(1, 1),padding="SAME", 
                     activation=tf.nn.relu,kernel_regularizer=tf.keras.regularizers.l2(l=1e-3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides=1,padding="SAME"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(l=1e-3)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10,kernel_regularizer=tf.keras.regularizers.l2(l=1e-3))
]) 


model1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

####### 4. Train model1

model1.fit(train_images, train_labels, epochs=10)

####### 5. Test model1

test_loss, test_acc = model1.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)