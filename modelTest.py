import torch
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import cv2
import os
import time

model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/IljaGavrylov/code/NematoVis/yolov5/runs/train/exp30/weights/best.pt', force_reload=True)

model.conf = 0.5

directory = 'C:/Users/Ilja/Nextcloud/NematoVis/Embryo-Index/'
subdirs = []
for subdir in os.listdir(directory):
    subdirs.append(directory + subdir)
print(subdirs)

for dir in subdirs:
    summaryFile = dir + '/summary.txt'
    #f = open(summaryFile, 'x')
    for file in os.listdir(dir):
        if file.lower().endswith(('.jpg')):
            #f = open(summaryFile, "a")
            img = os.path.join(dir, file)
            results = model(img)
            #f.write(file)
            #f.write(str(results.pandas().xyxy[0].value_counts('name')))
            #f.write('\n\n')
            #f.close()
            labeledImg = Image.fromarray(np.squeeze(results.render()), 'RGB')
            labeledImg = labeledImg.save(dir + '/' + file + '_labels.jpg')