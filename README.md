# RPSIC (Rock Paper Scissors Image Classifier)
For STEAM 2022 Science Fair :D
<br><br>
## WHAT IS THIS?

Hi! Welcome to Tristan, Steve and Louis's STEAM science fair project for 2022, which is a Rock Paper Scissors Image Classifier, or RPSIC. This README file will be dedicated to explaining the code we used for our project, and we hope that this explanation is understandable for you!
<br><br>
## TRAIN.PY

The purpose of the Python 3 file `train.py` is to set up, train and save a model, which will be saved to ```RPSModel.h5```

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
This function will take in an array of numbers `img` and a label `label`. It then turns the numbers into `float32` numbers and divides every number in `img` by 255. This is because `img` will be image data, which is from 0 to 255, and we 'shrink' it down into 0-1 to normalize it, so we divide it by 255. Note that `normalize` leaves `label` as is

Now we get to our dataset. First we have to retrieve our dataset:
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

`tfds.load()` loads a dataset from `tensorflow_datasets`. First, we have to specify what dataset we are using, which is `rock_paper_scissors`. Next, we can add in some extra keyword arguments (`kwargs`):
- `split` the dataset into `train` and `test`
- `shuffle_files` is set to True to randomly shuffle the items in the dataset
- `as_supervised` is set to True as we are going to use training and testing labels as a kind of 'answer sheet' to the images the model will be training on
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
Using `tf.data.experimental.AUTOTUNE` basically means we let tensorflow decide stuff for us. All you need to know about this section is that we are 'preparing' the data for the model to see and try to categorize later on

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
We are using a `Sequential` model, meaning each layer is placed one after another in a definite order. We have an `InputLayer` with shape of (300, 300, 3). This is because the images in our dataset are 300x300 and every pixel is divided into 3 color channels (r, g, b). Here are the other layers used:
- `Conv2D`, or Convolutional 2D layer, performs some calculation over each pixel in whatever input it gets, acting like a 'filter'
- `MaxPooling2D` layers downsample its inputs, so less calculations need to be made
- `Flatten` layers flatten its input into a 1-dimensional array (For example: ```[[1, 2], [3, 4, 5]] --> [1, 2, 3, 4, 5]```)
- `Dropout` layers randomly set some of its input into 0s to prevent overfitting
- `Dense` layers have each of its nodes connected to every node of the next layer, which is why they are called `Dense` layers

Note that every layer (aside `InputLayer`, `Flatten` and `Dropout`) has an activation function. In our model we are using 2 different functions:
- `ReLU` activation, or 'Rectified Linear Unit' activation, turns every negative value into a 0
- `Softmax` activation normalizes its input into a list of probabilities, which is why we are using it in the final layer of our model<br><br>

### **More About Softmax**

The Softmax activation function <img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;{\color{Teal}\sigma:&space;\mathbb{R}^x&space;\mapsto&space;[0,&space;1]^x}" title="{\color{Teal}\sigma: \mathbb{R}^x \mapsto [0, 1]^x}" /> is defined as:

<img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;{\color{Teal}\sigma(z)_i&space;=&space;\frac{e^{z_i}}{\sum_{k=1}^{x}e^{z_k}}}" title="{\color{Teal}\sigma(z)_i = \frac{e^{z_i}}{\sum_{k=1}^{x}e^{z_k}}}" />

for <img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;{\color{Teal}i=1,&space;2,&space;...,&space;x}" title="{\color{Teal}i=1, 2, ..., x}" /> and <img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;{\color{Teal}z&space;=&space;\left&space;\{z_1...z_k\right\}\in&space;\mathbb{R}^x}" title="{\color{Teal}z = \left \{z_1...z_k\right\}\in \mathbb{R}^x}" />. Note that the above function can also be expressed by the following:

<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;{\color{Teal}&space;\sigma(z)_i=\frac{e^{z_i}}{e^{z_1}&plus;e^{z_2}&plus;...&plus;e^{z_x}}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;{\color{Teal}&space;\sigma(z)_i=\frac{e^{z_i}}{e^{z_1}&plus;e^{z_2}&plus;...&plus;e^{z_x}}}" title="{\color{Teal} \sigma(z)_i=\frac{e^{z_i}}{e^{z_1}+e^{z_2}+...+e^{z_x}}}" /></a>
<br><br><br>

> ### **ReLU graph**
> 
> <img src="https://www.researchgate.net/publication/341158371/figure/fig4/AS:887822487674882@1588684784520/Rectified-linear-unit-ReLU-activation-function.ppm" alt="ReLU graph" width="300">

<br>

> ### **Softmax graph**
> 
> <img src="https://www.kindpng.com/picc/m/454-4548627_softmax-activation-function-hd-png-download.png" alt="Softmax graph" width="300">

<br>

Finally, lets train, test and save our model:
```python
model.compile(
  optimizer = 'adam',
  loss = 'sparse_categorical_crossentropy',
  metrics = ['accuracy'] 
)

model.fit(train_data, epochs = 5)
model.evaluate(test_data)
```
After we created the model, we have to compile the model with an optimizer algorithm, a loss function and a list of metrics to keep track of while the model is training:
- `adam` is one of, if not the best, adaptive optimizer algorithm, which is why we are using it
- `sparse_categorical_crossentropy` loss function will be used as our dataset is categorized into mutually exclusive categories (R, P, S)
- The `accuracy` metric is the only metric we will have to keep track of in this case

<br>

### **More About Sparse Categorial Crossentropy**

Categorial Crossentropy (CCE) is distinct from Sparse Categorical Crossentopy (SCCE) as the latter is used when our categories are mutually exclusive. The formula for CCE is defined as:

<img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;{\color{Teal}&space;Loss=-\sum_{k=1}^{C}p_k\log(q_k)}" title="{\color{Teal} Loss=-\sum_{k=1}^{C}p_k\log(q_k)}" />

where <img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;{\color{Teal}&space;C}" title="{\color{Teal} C}" /> is the output size, <img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;{\color{Teal}&space;p_k}" title="{\color{Teal} p_k}" /> is the <img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;{\color{Teal}&space;k}" title="{\color{Teal} k}" />th value in the output and <img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;{\color{Teal}&space;q_k}" title="{\color{Teal} q_k}" /> is the corresponding target value. Note that the above function can also be expressed by the following:

<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;{\color{Teal}&space;Loss=-p_1\log{q_1}-p_2\log{q_2}-...-p_C\log{q_C}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;{\color{Teal}&space;Loss=-p_1\log{q_1}-p_2\log{q_2}-...-p_C\log{q_C}}" title="{\color{Teal} Loss=-p_1\log{q_1}-p_2\log{q_2}-...-p_C\log{q_C}}" /></a>

<br>

Finally, we use `model.fit()` to start training our model. In this case we are using the `train_data` to train the model, and we are going to train the model over 5 epochs. After training is done, we run a test with data our model has not seen (`test_data`) to see how it does. After this test is done and its final accuracy is shown, the last 2 lines prompt the user to click enter to save the model to `C:\Users\trist\Desktop\Code\Python 3.x\STEAM2022\RPSModel.h5`:
```python
input("CLICK ENTER TO SAVE MODEL > ")
model.save(r'C:\Users\trist\Desktop\Code\Python 3.x\STEAM2022\RPSModel.h5')
```
<br><br>

### **CHANGES IN THE MODEL**

As you have seen from our explanation, our original model will have 5 epochs of training. We found that this overfitted our midel, leading to a not-so-good 66% final accuracy. To 'fix' this, we set up another training session, this time with 3 epochs instead of 5. This is so that the model wont get so 'used' to dealing with the images in our train dataset
