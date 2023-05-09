import torch
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import cv2
import os
import time

model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/Ilja/Nextcloud/NematoVis/test3/best.pt', force_reload=True)

model.conf = 0.5

#directory = 'C:/Users/Ilja/Nextcloud/NematoVis/test2/'
# subdirs = []
# for subdir in os.listdir(directory):
#     subdirs.append(directory + subdir)
# print(subdirs)
###
subdirs = ['C:/Users/Ilja/Nextcloud/NematoVis/test3/']

for dir in subdirs:
    # create or overwrite a summary file
    summaryFile = dir + 'summary.txt'
    if os.path.isfile(summaryFile):
       f = open(summaryFile, 'w')
    else: 
        f = open(summaryFile, 'x')
    
    # delete all previous result images
    filesInDir = os.listdir(dir)
    for file in filesInDir:
        if file.endswith("_labels.jpg"):
            os.remove(os.path.join(dir, file))

    # perform detection on all images, write results to summary file, save result images
    for file in os.listdir(dir):
        if file.lower().endswith(('.png')) or file.lower().endswith(('.jpg')):
            f = open(summaryFile, "a")
            img = os.path.join(dir, file)
            results = model(img)
            f.write(file)
            f.write(str(results.pandas().xyxy[0].value_counts('name')))
            f.write('\n\n')
            f.close()
            labeledImg = Image.fromarray(np.squeeze(results.render()), 'RGB')
            labeledImg = labeledImg.save(dir + '/' + file + '_labels.jpg')