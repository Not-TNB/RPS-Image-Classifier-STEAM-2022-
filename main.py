from tkinter import *
from PIL import ImageTk, Image
import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model(r'C:\Users\trist\Desktop\Code\Python 3.x\STEAM2022\RPSModel.h5')

size = 300, 300
def transform(path):
  im = Image.open(path)
  w, h = im.size
  if h != w: return None
  im.resize(size)
  img = im.load()
  data = [img[x, y] for y in range(300) for x in range(300)]
  return tf.cast(data, tf.float32)/255.0

DARK = '#082032'
VERY_DARK = '#2C394B'
KINDA_DARK = '#334756'
ORANGE = '#FF4C29'

root = Tk(className = '(STEAM 2022) RPS Hand Pose Image Classifier')
root.geometry("800x700")
root.configure(bg = DARK)

pathfield = Entry(root, width = 50, bg = VERY_DARK, fg = ORANGE, bd = 0, borderwidth = 5, relief = FLAT, font = "Helvetica 14")
pathfield.insert(0, "Enter the path to a 1:1 image :D")
pathfield.grid(row = 0, column = 0, padx = 30, pady = 30, ipadx = 8, ipady = 5)

img = Label(root, image = ImageTk.PhotoImage(Image.new('RGB', (300, 300))), borderwidth = 10)
img.grid(row = 1, column = 0, padx = 10, pady = 30)

predict = Label(root, text = "Rock: _\nPaper: _\nScissors: _", font = "Helvetica 20", bg = DARK, fg = ORANGE)
predict.grid(row = 2, column = 0, padx = 10, pady = 30)

def process(path):
  im = ImageTk.PhotoImage(Image.open(path))
  img.configure(image = im)
  img.image = im
  data = transform(path)
  data = tf.reshape(data, [1, 300, 300, 3])
  prediction = model.predict(data)
  predict.configure(text = f"Rock: {int(prediction[0][0]*100)}%\nPaper: {int(prediction[0][1]*100)}%\nScissors: {int(prediction[0][2]*100)}%")

e = Button(root, text = 'Enter', padx = 30, pady = 20, bg = VERY_DARK, fg = ORANGE, command = lambda: process(pathfield.get()), activeforeground = KINDA_DARK, bd = 0, font = "Helvetica 14")
e.grid(row = 0, column = 1, padx = 10, pady = 30)

root.mainloop()
