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
