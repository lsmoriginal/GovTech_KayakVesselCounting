# given a particular set of training data, this script visualises the 
# bouding box style of the given training data

# this is a helper tool to visualise the train/valid dataset
# this script is also use to crop out the vessels/kayak
# to use the crops in augmentation


import sys
# sys.path.append('/hpctmp/e0310593/Lecture1_Assignment/depends')

import yaml 
import argparse
from tqdm import tqdm
import os
import pandas as pd
from time import time

import cv2

from utils.plots import Annotator, colors
from utils.general import xywhn2xyxy, save_one_box

class DrawBox():
    
    def __init__(self, config):
        
        self.config = config
        
        if os.path.isdir(config.images) and os.path.isdir(config.labels):
            # if both are folders
            self.images = tuple(fileName for fileName in os.listdir(config.images) if fileName.endswith('.jpg'))
            self.labels = set(fileName for fileName in os.listdir(config.labels) if fileName.endswith('.txt'))
            
        # elif os.path.exists(config.images) and os.path.exists(config.labels):
            # both are single file 
            
            # self.images = (os.path.basename(config.images),)
            # self.labels = (os.path.basename(config.labels),)
            # self.config.images = os.path.dirname(self.config.images)
            # self.config.labels = os.path.dirname(self.config.labels)
        else:
            raise Exception('Must both be directory or both be a single file')
        
        # if os.path.exists(config.dest) and os.path.isdir(config.dest):
        if False:
            self.dest = config.dest
        else:
            self.dest = os.path.join(config.images, 'labelled')
            os.makedirs(self.dest) if not os.path.exists(self.dest) else None
            
        if self.config.save_crop:
            self.cropDest = os.path.join(self.dest, 'Cropped')
            os.makedirs(self.cropDest) if not os.path.exists(self.cropDest) else None
            
        
    def draw(self):
        progressBar = tqdm(self.images, desc='Progress')
        timeiter, iterCount = time(), 0
        for eachImage in self.images:
            labelName = eachImage.replace('jpg', 'txt')
            if labelName not in self.labels:
                continue
            
            image = cv2.imread(os.path.join(self.config.images, eachImage))
            labels = pd.read_csv(os.path.join(self.config.labels, labelName), 
                                 sep='\s+', header=None)
            
            self.drawOneImage(image, labels, eachImage)
            cv2.imwrite(os.path.join(self.dest, eachImage),
                        image)
            
            iterCount += 1
            if time() - timeiter > 2:
                progressBar.update(iterCount)
                iterCount = 0
            
            
        
    
    def drawOneImage(self, image, labels,imageName):
        annotator = Annotator(image, 
                              line_width=self.config.line_thickness)
        labels.iloc[:, 0] = labels.iloc[:, 0].apply(str)
        
        height, width, colorLayers = image.shape
        xywhs = labels.iloc[:,1:5].to_numpy()
        xyxys = xywhn2xyxy(xywhs, width, height)
        imageNameBase = imageName.replace('.jpg', '')
        if self.config.save_crop:
            for index, xyxy in enumerate(xyxys):
                cropName = "_".join((imageNameBase, str(index), labels.iloc[index, 0],'.jpg'))
                save_one_box(xyxy, image,  
                             file= os.path.join(self.cropDest, cropName),
                             BGR=True)  
        
        for index, xyxy in enumerate(xyxys):
            annotator.box_label(xyxy, labels.iloc[index, 0], 
                                color=colors(labels.iloc[index, 0], True))    
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=str, default='./')
    parser.add_argument('--labels', type=str, default='./')
    parser.add_argument('--dest', type=str, default='./labelledIMages')
    
    parser.add_argument('--save-crop', action='store_true')
    parser.add_argument('--line-thickness', type=float, default=2)
    
    opt = parser.parse_args()
    
    boxPainter = DrawBox(opt)
    boxPainter.draw()
    
    # python3 ./drawBoxes.py --images ../Data/Vessel_Kayak_Count/images/images/train --labels ../Data/Vessel_Kayak_Count/images/labels/train --save-crop
    # python3 ./drawBoxes.py --images ../Data/annotated_1/obj_train_data --labels ../Data/annotated_1/obj_train_data --save-crop
    # python3 ./drawBoxes.py --images ../Data/videoData/obj_train_data --labels ../Data/videoData/obj_train_data --save-crop
    # python3 ./drawBoxes.py --images ../Data/augmentedData/images/train --labels ../Data/augmentedData/labels/train
    
    # python3 ./drawBoxes.py --images ../Data/fullyAugmented/data/images/train --labels ../Data/fullyAugmented/data/labels/train
    
    # python3 ./drawBoxes.py --images ../Data/agumentedData_oriMixNew_smoothing_brightness_rain_argparseTest/images/train --labels ../Data/agumentedData_oriMixNew_smoothing_brightness_rain_argparseTest/labels/train