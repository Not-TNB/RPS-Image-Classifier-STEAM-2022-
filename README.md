# RPS Image Classifier
For STEAM 2022 Science Fair :D
<br><br>
## CODE EXPLANATION
### train.py
The purpose of `train.py` is to set up, train and save a model, which will be saved to ```RPSModel.h5```

The first 4 lines of `train.py` imports `numpy`, `keras`, `tensorflow` and `tensorflow_datasets`:
```python
import numpy as np
from tensorflow import keras
import tensorflow as tf
import tensorflow_datasets as tfds
```
- `numpy` offers many mathematical functions, and we are using it here for its arrays
- `keras` is used for Artifical Intelligence, and we are using it to help create and train our model
- `tensorflow` is used for Machine Learning, and we are going to use it alongside `keras` to make a working model
- `tensorflow_datasets` is a collection of convenient datasets, and this is where we are going to get our RPS dataset

Next, we have a `normalize` function:
```python
def normalize(img, label): return tf.cast(img, tf.float32)/255.0, label
```
This function will take in an array of numbers `img` and a label `label`. It then turns the numbers into `float32` numbers and divides every number in `img` by 255. This is because `img` will be image data, which is from 0 to 255, and we 'shrink' it down into 0-1 to normalize it, so we divide it by 255

Now we get to our dataset. First we take our model:
```python
(train_data, test_data), dsinfo = tfds.load(
  'rock_paper_scissors',
  split = ['train', 'test'],
  shuffle_files = True,
  as_supervised = True,
  with_info = True
)
```
We will split our data into `train_data` and `test_data`, and we will also get information about the dataset `dsinfo` (`dsinfo` is not needed for training its just good to understand our data)

`tfds.load()` loads a dataset from `tensorflow_datasets`. First, we have to specify what dataset we are using, which is `rock_paper_scissors`. Next, we can add in some extra arguments:
- `split` the dataset into `train` and `test`
- `shuffle_files` is set to True to randomly shuffle the items in the dataset
- `as_supervised` is set to True as we are going to use `labels` as a kind of 'answer sheet' to the images the model will be training on
- `with_info` is set to True so we can get the dataset info

Now that we have our dataset, we can get to preprocessing our dataset:
```python
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_data = train_data.map(normalize, num_parallel_calls = AUTOTUNE)
train_data = train_data.cache()
train_data = train_data.shuffle(dsinfo.splits['train'].num_examples)
train_data = train_data.batch(2)
train_data = train_data.prefetch(AUTOTUNE)

test_data = test_data.map(normalize, num_parallel_calls = AUTOTUNE)
test_data = test_data.batch(4)
test_data = test_data.prefetch(AUTOTUNE)
```
Using `tf.data.experimental.AUTOTUNE` basically means we let tensorflow decide stuff for us
All you need to know about this section is that we are 'preparing' the data for the model to see and try to categorize later on

Let's now get to making our model 'structure':
```python
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
```
We are using a `Sequential` model, meaning each layer is 'piled up' next to each other. We have an `InputLayer` with shape of (300, 300, 3). This is because the images in our dataset are 300x300 and every pixel is divided into 3 color channels (r, g, b)
- `Conv2D`, or Convolutional 2D layer, performs some calculation over each pixel in whatever input it gets, acting like a 'filter'
- `MaxPooling2D` layers downsample its inputs, so less calculations need to be made
- `Flatten` layers flatten its input into a 1-dimensional array (For example: ```[[1, 2], [3, 4, 5]] --> [1, 2, 3, 4, 5]```)
- `Dropout` layers randomly set some of its input into 0s to prevent overfitting
- `Dense` layers have each of its nodes connected to every node of the next layer, which is why they are called `Dense` layers

Note that every layer (aside `InputLayer`, `Flatten` and `Dropout`) has an activation function. In our model we are using 2 different functions:
- `ReLU` activation, or 'Rectified Linear Unit' activation, turns every negative value into a 0
- `Softmax` activation normalizes its input into a list of probabilities, which is why we are using it in the final layer of our model
