import os, json, base64, cv2, glob
import numpy as np
import matplotlib.pyplot as plt
from coco import CocoConfig
from Mask.config import Config
import Mask.utils as utils
import Mask.model as modellib
import Mask.visualize as visualize
from convert_file import load_image

def init():
    np.set_printoptions(threshold=np.inf)

def run(input_df):
    data = json.loads(input_df)
    im = load_image(image64=data)

    config = CocoConfig()

    model = modellib.MaskRCNN(mode="inference", model_dir="./models/", config=config)
    model.load_weights(filepath="./models/mask_rcnn_moles_0090.h5", by_name=True)

    class_names = ["BG", "malignant", "benign"]

    # predict the mask, bounding box and class of the image
    r = model.detect([im])[0]
    prediction = None
    for idx, val in enumerate(class_names):
        if idx == r["class_ids"]:
            prediction = val
            print(val)
        else:
            continue
    return prediction