import os
import re
import pandas as pd

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
#from keras.applications import SqueezeNet
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model


def BuildDataframe(directory):
    data = {'image_path': [], 'label': []}
    pattern = r'diseased'

    for plant in os.listdir(directory):
        class_dir = os.path.join(directory, plant)

        if os.path.isdir(class_dir):
            match = re.search(pattern, plant)
            label = "healthy"
            if match:
                label = "diseased"
                    
            print("Extracting ", plant, "...")            
            for image_file in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_file)
                data['image_path'].append(image_path)
                data['label'].append(label)

    print("Data correctly extracted.")
    dataframe = pd.DataFrame(data)
    return dataframe

train_dir = '/Users/guilhem/Desktop/Projet HAEEAI/input/plant-leaves-for-image-classification/train'
train_dataframe = BuildDataframe(train_dir)

test_dir = '/Users/guilhem/Desktop/Projet HAEEAI/input/plant-leaves-for-image-classification/test'
test_dataframe = BuildDataframe(test_dir)

valid_dir = '/Users/guilhem/Desktop/Projet HAEEAI/input/plant-leaves-for-image-classification/valid'
valid_dataframe = BuildDataframe(valid_dir)

sample_df = train_dataframe.sample(n=10)
print(sample_df)

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("GPU config OK")
else:
    print("Error using GPU, CPU will be used")
tf.keras.backend.clear_session()

input_shape = (224, 224, 3)
num_classes = 2
batch_size = 32
epochs = 10

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_directory = '/Users/guilhem/Desktop/Projet HAEEAI/input/plant-leaves-for-image-classification/train'
test_directory = '/Users/guilhem/Desktop/Projet HAEEAI/input/plant-leaves-for-image-classification/test'
valid_directory = '/Users/guilhem/Desktop/Projet HAEEAI/input/plant-leaves-for-image-classification/valid'

print(train_dataframe['image_path'][0])

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_dataframe,
    directory=train_directory,
    x_col='image_path',
    y_col='label',
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_dataframe,
    directory=test_directory,
    x_col='image_path',
    y_col='label',
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical'
)

valid_generator = test_datagen.flow_from_dataframe(
    dataframe=valid_dataframe,
    directory=valid_directory,
    x_col='image_path',
    y_col='label',
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical'
)

input_shape = (224, 224, 3)
num_classes = 2

model = tf.keras.Sequential()
model.add(Conv2D(2, (2, 2), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_generator, epochs=epochs, validation_data=valid_generator)

model.save('../model_saved/modelquantif3DENSE.h5')

test_loss, test_acc = model.evaluate(test_generator)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)

model.summary()
