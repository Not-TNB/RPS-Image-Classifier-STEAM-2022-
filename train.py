import numpy as np
from tensorflow import keras
import tensorflow as tf
import tensorflow_datasets as tfds

def normalize(img, label): return tf.cast(img, tf.float32)/255.0, label

(train_data, test_data), dsinfo = tfds.load(
  'rock_paper_scissors',
  split = ['train', 'test'],
  shuffle_files = True,
  as_supervised = True,
  with_info = True
)

AUTOTUNE = tf.data.experimental.AUTOTUNE
train_data = train_data.map(normalize, num_parallel_calls = AUTOTUNE)
train_data = train_data.cache()
train_data = train_data.shuffle(dsinfo.splits['train'].num_examples)
train_data = train_data.batch(2)
train_data = train_data.prefetch(AUTOTUNE)

test_data = test_data.map(normalize, num_parallel_calls = AUTOTUNE)
test_data = test_data.batch(4)
test_data = test_data.prefetch(AUTOTUNE)

model = keras.Sequential([
  keras.layers.InputLayer((300, 300, 3)),
  keras.layers.Conv2D(400, 3, activation = 'relu'),
  keras.layers.MaxPooling2D(2, 2),
  keras.layers.Conv2D(400, 3, activation = 'relu'),
  keras.layers.MaxPooling2D(2, 2),
  keras.layers.Conv2D(400, 3, activation = 'relu'),
  keras.layers.MaxPooling2D(2, 2),
  keras.layers.Flatten(),
  keras.layers.Dropout(0.5),
  keras.layers.Dense(750, activation = 'relu'),
  keras.layers.Dense(3, activation = 'softmax')
])

model.compile(
  optimizer = 'adam',
  loss = 'sparse_categorical_crossentropy',
  metrics = ['accuracy'] 
)

model.fit(train_data, epochs = 3)
model.evaluate(test_data)

input("CLICK ENTER TO SAVE MODEL > ")
model.save(r'RPSModel.h5')
