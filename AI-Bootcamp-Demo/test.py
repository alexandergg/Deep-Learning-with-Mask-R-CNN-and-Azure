import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
import glob
from Mask.config import Config
import Mask.utils as utils
import Mask.model as modellib
import Mask.visualize as visualize
from coco import CocoConfig

np.set_printoptions(threshold=np.inf)

dir_path = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = dir_path + "/models/"
MODEL_PATH = input("Insert the path of your trained model [ Like models/moles.../mask_rcnn_moles_0090.h5 ]: ")
if os.path.isfile(MODEL_PATH) == False:
    raise Exception(MODEL_PATH + " Does not exists")

path_data = input("Insert the path of Data [ Link /home/../ISIC-Archive-Downloader/Data/ ] : ")
if not os.path.exists(path_data):
    raise Exception(path_data + " Does not exists")

config = CocoConfig()

model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
model.load_weights(MODEL_PATH, by_name=True)

class_names = ["BG", "malignant", "benign"]

all_desc_path = glob.glob(path_data + "Descriptions/ISIC_*")
for filename in os.listdir(path_data+"Descriptions/"):
    data = json.load(open(path_data+"/Descriptions/"+filename))
    img = cv2.imread(path_data+"Images/"+filename+".jpeg")
    img = cv2.resize(img, (128, 128))

    ## ground truth of the class
    print(data["meta"]["clinical"]["benign_malignant"])
    
    r = model.detect([img])[0]
    visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'],
                                class_names, r['scores'])