from tkinter import *
from PIL import ImageTk, Image
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2

model = keras.models.load_model(r'RPSModel.h5')

size = 300, 300
def transform(path):
  im = Image.open(path)
  w, h = im.size
  if h != w: return None
  im.resize(size)
  img = im.load()
  data = [img[x, y] for y in range(300) for x in range(300)]
  return tf.cast(data, tf.float32)/255.0

DARK = '#334257'
DARK_BLUE = '#476072'
BLUE = '#548CA8'
WHITE = '#EEEEEE'

root = Tk(className = '(STEAM 2022) RPS Hand Pose Image Classifier')
root.geometry('880x700')
root.resizable(False, False)
root.configure(bg = DARK)

pathfield = Entry(root, width = 50, bg = DARK_BLUE, fg = WHITE, bd = 0, borderwidth = 5, relief = FLAT, font = 'Helvetica 14')
pathfield.insert(0, 'Enter the path to a 1:1 image :D')
pathfield.grid(row = 0, column = 0, padx = 30, pady = 30, ipadx = 8, ipady = 5)

img = Label(root, image = ImageTk.PhotoImage(Image.new('RGB', (300, 300))), borderwidth = 10)
img.grid(row = 1, column = 0, padx = 10, pady = 30)

predict = Label(root, text = 'Rock: _\nPaper: _\nScissors: _', font = 'Helvetica 20', bg = DARK, fg = WHITE)
predict.grid(row = 2, column = 0, padx = 10, pady = 30)

def process(path):
  im = ImageTk.PhotoImage(Image.open(path).resize(size))
  img.configure(image = im)
  img.image = im
  data = transform(path)
  data = tf.reshape(data, [1, 300, 300, 3])
  prediction = model.predict(data)
  predict.configure(text = f'Rock: {int(prediction[0][0]*100)}%\nPaper: {int(prediction[0][1]*100)}%\nScissors: {int(prediction[0][2]*100)}%')

def camon():
  cam = cv2.VideoCapture(0)
  cam.set(3, 1280)
  cam.set(4, 720)
  r = 0
  p = 0
  s = 0
  guess = "None"
  while True:
    success, img = cam.read()
    rect = cv2.rectangle(img, (100, 100), (450, 450), (255, 255, 255), 2)
    cv2.putText(img, 'Put your hand in the white frame!', (470, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, 'Q: Quit Webcam Mode', (470, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, 'W: Predict using image in frame', (470, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, '(Holding W is laggier but its better :L)', (470, 270), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
    if cv2.waitKey(1) & 0xFF == ord('w'): 
      x = img[100:450, 100:450]
      im = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
      im = cv2.resize(im, size)
      prediction = model.predict(np.array([im]))
      r, p, s = [n for n in prediction[0]]
      if r == 1.0: guess = "Rock"
      elif p == 1.0: guess = "Paper"
      else: guess = "Scissors"
    cv2.putText(img, f'Prediction: {guess}', (100, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('Live Camera', img)    
  cam.release()
  cv2.destroyAllWindows()

e = Button(root, text = 'ENTER', padx = 30, pady = 20, bg = DARK_BLUE, fg = WHITE, command = lambda: process(pathfield.get()), activeforeground = BLUE, bd = 0, font = 'Helvetica 14')
e.grid(row = 0, column = 1, padx = 10, pady = 30)

togglecam = Button(root, text = 'Webcam Mode', padx = 30, pady = 20, bg = DARK_BLUE, fg = WHITE, command = camon, activeforeground = BLUE, bd = 0, font = 'Helvetica 14')
togglecam.grid(row = 1, column = 1, padx = 10, pady = 30)

root.mainloop()
