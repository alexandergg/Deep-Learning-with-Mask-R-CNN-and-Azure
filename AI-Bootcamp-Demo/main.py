#!/usr/bin/env python
from flask import Flask, request
from flask_cors import CORS, cross_origin
import os
import json
import urllib
import numpy as np
from convert_file import load_image
import Mask.visualize as visualize
from Mask.config import Config
import Mask.utils as utils
import Mask.model as modellib
from coco import CocoConfig
import matplotlib.pyplot as plt
import cv2
import PIL
import uuid
from keras import backend as K

class NumpyEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, np.integer): return int(obj)
    elif isinstance(obj, np.floating): return float(obj)
    elif isinstance(obj, np.ndarray): return obj.tolist()
    else: return super(NumpyEncoder, self).default(obj)

def classify(rgb_img):

  config = CocoConfig()

  model = modellib.MaskRCNN(mode="inference", model_dir="./models/", config=config)
  model.load_weights(filepath="./models/mask_rcnn_moles_0030.h5", by_name=True)

  class_names = ['Salmonella','Yersinia','Shigella','Negativo']
  r = model.detect([rgb_img])[0]

  image, caption = visualize.display_instances(rgb_img, r['rois'], r['masks'], r['class_ids'],
                              class_names, r['scores'])
  K.clear_session()
  image = PIL.Image.fromarray(image, 'RGB')
  words = caption.split()
  filename = str(uuid.uuid4())
  filepath = "./static/img/test/{}.jpg".format(filename)
  image.save(filepath)

  return json.dumps([{"caption": words[0],
                      "percent": words[1],
                      "image_path": filepath}], cls=NumpyEncoder)

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
  return app.send_static_file('index.html')

@app.route('/js/<path:path>')
def send_js(path):
  return send_from_directory('js', path)

@app.route('/css/<path:path>')
def send_css(path):
  return send_from_directory('css', path)

@app.route('/api/v1/classify', methods=['POST'])
def classifyOnPost():
  filestr = request.files['file'].read()
  if filestr:
    npimg = np.fromstring(filestr, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return classify(rgb_img)

if __name__ == "__main__":
  app.run(host='0.0.0.0')
