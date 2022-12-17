
# import flask


from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename

#import model

import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import random 
import tensorflow as tf
import cv2
from segmentation_models.metrics import iou_score
from segmentation_models import Unet
import segmentation_models as sm
import skimage.transform as trans
import base64
import matplotlib.pyplot as plt
sm.set_framework('tf.keras')
sm.framework()

w,h = 256,256
batch_size = 16
BACKBONE = "resnet34"
preprocess_input = sm.get_preprocessing(BACKBONE)

def read_image(image_path):
  image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
  image = image / 255
  image = trans.resize(image,(256,256,3))
  image = preprocess_input(image)
  return image
def read_mask(image_path):
  image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
  image = image /255
  image = trans.resize(image,(256,256,3))
  image[image != 0] = 1
  return image


app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads/"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return render_template('index.html')


@app.route("/",methods = ['POST','GET'])
def upload_image():
  if 'file' not in request.files:
    
    return redirect(request.url)
  file = request.files['file']
  if file.filename == '':
      
      return redirect(request.url)
  if file and allowed_file(file.filename):
      filename = secure_filename(file.filename)
      file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

      #predict
      model = Unet(BACKBONE,encoder_weights='imagenet',classes=1,input_shape=(256,256,3),activation='sigmoid',encoder_freeze=True)
      model.load_weights("point.hdf5")
      x_test = read_image(os.path.join(app.config['UPLOAD_FOLDER'], filename))
      x_test = np.array(x_test).astype('float32')
      x_test = x_test[np.newaxis,:,:,:]
      y_pre = model.predict(x_test)
      y_pre = trans.resize(y_pre[0],(256,256,3))

      plt.imshow(y_pre)
      path = filename.replace(".jpg",".png")
      path = UPLOAD_FOLDER + path
      plt.savefig(path)

      return render_template('index.html', filename=filename)
  else:
      flash('Allowed image types are - png, jpg, jpeg, gif')
      return redirect(request.url)
  
@app.route('/display/<filename>')
def display_image(filename):
    path = str(filename).replace('.jpg','.png')
    return redirect(url_for('static', filename='uploads/' + path), code=301)

@app.route('/show/<filename>')
def show_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)
if __name__ == "__main__":
    app.run(port=9050)