# scenario2_image-recognition.py

import numpy as np
import tensorflow as tf
from tensorflow import keras
import boto3
import pickle

# your-bucket-name
# └───cifar-10-batches-py
#     ├── batches.meta
#     ├── data_batch_1
#     ├── data_batch_2
#     ├── data_batch_3
#     ├── data_batch_4
#     ├── data_batch_5
#     ├── readme.html
#     └── test_batch
#
# TODO: Change the bucket name
bucket_name = 'your-bucket-name'


s3 = boto3.client('s3')

# Load the dataset from S3
def load_data(bucket_name, file_name):
    obj = s3.get_object(Bucket=bucket_name, Key=file_name)
    data = obj['Body'].read()
    return pickle.loads(data, encoding='bytes')

train_data = np.empty((0, 32*32*3))
train_labels = []

# Load 5 batches of training data and one batch of test data from the S3 bucket
for i in range(1, 6):
    batch = load_data(bucket_name, 'cifar-10-batches-py/data_batch_' + str(i))
    train_data = np.vstack((train_data, batch[b'data']))
    train_labels += batch[b'labels']

train_labels = np.array(train_labels)
test_data = load_data(bucket_name, 'cifar-10-batches-py/test_batch')[b'data']
test_labels = np.array(load_data(bucket_name, 'cifar-10-batches-py/test_batch')[b'labels'])

# Reshape and normalize the data
train_data = train_data.reshape(-1, 32, 32, 3) / 255.0
test_data = test_data.reshape(-1, 32, 32, 3) / 255.0

# Define the model
model = keras.Sequential([
    keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(32, 32, 3)),
    keras.layers.MaxPooling2D(pool_size=2),
    keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
    keras.layers.MaxPooling2D(pool_size=2),
    keras.layers.Flatten(),
    keras.layers.Dense(units=128, activation='relu'),
    keras.layers.Dense(units=10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_data, train_labels, epochs=10, batch_size=32, validation_split=0.1)

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(test_data, test_labels)
print('Test accuracy:', test_acc)
