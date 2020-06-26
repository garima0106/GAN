from __future__ import print_function
import tensorflow.compat.v2 as tf
import sys 
import time
import argparse
import os
import json
import tensorflow_datasets as tfds
import tensorflow as tf


NUM_WORKERS = 1
GLOBAL_BATCH_SIZE = 64 * NUM_WORKERS
BUFFER_SIZE = 10000


def make_datasets_unbatched():
  # Scaling MNIST data from (0, 255] to (0., 1.]
  def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label

  datasets, info = tfds.load(name='mnist',
                            with_info=True,
                            as_supervised=True)

  return datasets['train'].map(scale).cache().shuffle(BUFFER_SIZE)

def build_and_compile_cnn_model():
  model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(10)
  ])
  model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
      metrics=['accuracy'])
  return model


def main(_):
 
  os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ["192.168.1.241:2222"],
        'ps' :['192.168.1.18:2222']
    },
    'task': {'type': 'worker', 'index': 0}
})
 
  strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
  #strategy = tf.distribute.experimental.ParameterServerStrategy()
  
  train_datasets = make_datasets_unbatched().batch(GLOBAL_BATCH_SIZE)
   
  callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath='/tmp/keras-ckpt')]
  with strategy.scope():
    multi_worker_model = build_and_compile_cnn_model()
  multi_worker_model.fit(x=train_datasets,
                       epochs=3,
                       steps_per_epoch=5,
                       callbacks=callbacks)

     
if __name__ == "__main__":

   tf.compat.v1.app.run(main=main)
  
