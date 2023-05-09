import torch
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import cv2
import os
import time

model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/IljaGavrylov/code/NematoVis/yolov5/runs/train/exp30/weights/best.pt', force_reload=True)

model.conf = 0.5  # NMS confidence threshold
#model.iou = 0.45  # NMS IoU threshold
#model.agnostic = False  # NMS class-agnostic
#model.multi_label = False  # NMS multiple labels per box
#model.classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
#model.max_det = 1000  # maximum number of detections per image
#model.amp = False  # Automatic Mixed Precision

img1 = os.path.join('C:/Users/Ilja/Nextcloud/NematoVis/Embryo-Index/w1', '50HD0001.jpg')
img2 = os.path.join('C:/Users/Ilja/Nextcloud/NematoVis/Embryo-Index/w1', '50HD0002.jpg')
img3 = os.path.join('C:/Users/Ilja/Nextcloud/NematoVis/Embryo-Index/w1', '50HD0003.jpg')

results1 = model(img1)
results2 = model(img2)
results3 = model(img3)

results1.print()
print(type(str(results1.pandas().xyxy[0].value_counts('name'))))
print(str(results1.pandas().xyxy[0].value_counts('name')))
results2.print()
results3.print()

img1 = Image.fromarray(np.squeeze(results1.render()), 'RGB')
#img1.show()

img2 = Image.fromarray(np.squeeze(results2.render()), 'RGB')
#img2.show()

img3 = Image.fromarray(np.squeeze(results3.render()), 'RGB')
#img3.show()
# plt.imshow(np.squeeze(results.render()))
# plt.show()