import tensorflow as tf
from tensorflow import lite

model = tf.keras.models.load_model('/Users/guilhem/Desktop/Projet HAEEAI/programme/model.h5')

converter = lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_quantized_model = converter.convert()

with open('model_quantized.tflite', 'wb') as f:
    f.write(tflite_quantized_model)