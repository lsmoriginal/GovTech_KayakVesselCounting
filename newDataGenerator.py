# this script will take in:
# a folder container water bodies
# a folder of kayak
# a folder of veseel

# Then generate a YOLO dataset
# by combining and permutating the images

import cv2
import numpy as np
import pandas as pd 
from pathlib import Path
from random import uniform
from random import choice
from random import randint
from utils.general import xyxy2xywhn
from tqdm import tqdm
from functools import partial
import multiprocessing as mp
from itertools import repeat
import argparse

def createMask(ones, windowSize):
    leftBound = windowSize
    rightBound = ones.shape[1]-windowSize
    upBound = windowSize
    downBound = ones.shape[0]-windowSize
    center = ones.shape[0]/2, ones.shape[1]/2
    
    for i in range(ones.shape[0]):
        for j in range(ones.shape[1]):
            if upBound < i < downBound and leftBound < j < rightBound:
                dist = ((abs(center[0] - i)/center[0])**2 + (abs(center[1] - j)/center[1])**2)**(1/2)
                dist = dist ** (1/2)
            else:
                # scaled in X axis
                dist = ((abs(center[0] - i)/center[0])**2 + (abs(center[1] - j)/center[1])**2)**(1/2)
                # print(dist)
            ones[i,j] = dist
    return (ones/ones.max())**1.2

def createMask(image):
    mask = np.ones(image.shape[:2])
    shape = mask.shape
    size = max(shape)
    center = size/2
    mask = cv2.resize(mask, (size, size))
    
    for i in range(size):
        for j in range(size):
            dist = ((i-center)**4 + (j-center)**4)**(1/4)
            if dist > center:
                mask[i,j] = 1
            else:
                mask[i,j] = (dist/center)
    return cv2.resize(mask, (shape[1], shape[0]))

def generate_random_lines(imshape,
                          slant,
                          drop_length,
                          rainIntensity):    
    drops=[]    
    for i in range(rainIntensity): ## If You want heavy rain, try increasing this        
        if slant<0:            
            x= np.random.randint(slant,imshape[1])        
        else:            
            x= np.random.randint(0,imshape[1]-slant)        
            y= np.random.randint(0,imshape[0]-drop_length)        
            drops.append((x,y))    
    return drops

class Image():
    
    def __init__(self, img):
        if isinstance(img, str):
            self.image = cv2.imread(img)
        elif isinstance(img, np.ndarray):
            self.image = img
        else:
            self.image = cv2.imread(str(img))
        if isinstance(self.image, type(None)):
            raise Exception(f'{img} not found!')
    
    def hFlip(self):
        return Image(self.image[:,::-1,:])
    
    def mixupWith(self, image2, 
                  mixup_ratio=0.3, 
                  X=0.5, Y=0.2,
                  smoothBorder = False):

        '''
        image1, image2: images to be mixed up, type=ndarray
        mixup_ratio: ratio in which two images are mixed up
        XY: location on image1 that the center of the image2 will be located at, normalised position
        Returns a mixed-up image with new set of smoothed labels
        '''
        # height = max(image1.shape[0], image2.shape[0])
        # width = max(image1.shape[1], image2.shape[1])
        # mix_img = np.zeros((height, width, 3),dtype=np.float32)
        # mix_img[:image1.shape[0], :image1.shape[1], :] = image1.astype(np.float32)\
        #                                                  * mixup_ratio
        image1 = self.image
        image2 = image2.image
        a,b = image1.shape, image2.shape
        
        mix_img = image1
        y_start = int(image1.shape[0] * Y -  0.5 * image2.shape[0])
        if y_start<0:
            yoff = abs(y_start)
            image2 = image2[yoff:,:]
            y_start=0
        y_end = y_start + image2.shape[0]
        if y_end>image1.shape[0]:
            yoff=y_end-image1.shape[0]
            image2 = image2[:-yoff,:]
            y_end = image1.shape[0]
        x_start = int(image1.shape[1] * X -  0.5 * image2.shape[1])
        if x_start<0:
            xoff = abs(x_start)
            image2 = image2[:,xoff:]
            x_start = 0
        x_end = x_start+image2.shape[1]
        if x_end>image1.shape[1]:
            xoff = x_end-image1.shape[1]
            image2 = image2[:,:-xoff]
            x_end=image1.shape[1]
        
        originalPlot = mix_img[y_start:y_end,x_start:x_end, :]
        
        if smoothBorder:
            # let the widow be about X% of the entire cropped ship
            # mask = np.ones(image2.shape[:2])
            # windowSize = int(min(mask.shape)*0.2)
            mask = createMask(image2)
            # print(mask)
            for i in range(3):
                originalPlot[:,:,i] = originalPlot[:,:,i] * (mask)
                image2[:,:,i] = image2[:,:,i] * (1- mask)
            mix_img[y_start:y_end,x_start:x_end, :] = originalPlot + image2
        else:
            try:
                mix_img[y_start:y_end,x_start:x_end, :] = originalPlot * mixup_ratio + image2 * (1-mixup_ratio)
            except ValueError:
                print(originalPlot.shape, image2.shape, (x_start, x_end), a,b, X,Y)
        # [y_start,y_end,x_start,x_end] 
        return Image(mix_img), [x_start,y_start, x_end,y_end]
        
    
    def addRandomColor(self):
        color1 = np.random.choice(range(256), size=3)
        image = self.image.copy()
        image = image + (np.ones_like(self.image) * color1)
        image = image.astype(np.uint8)
        return Image(image)
    
    def addBrightness(self, brightness):
        image = self.image.copy()
        image_HLS = cv2.cvtColor(image, cv2.COLOR_RGB2HLS) # Conversion to HLS    
        image_HLS = np.array(image_HLS, dtype = np.float64)
        
        # random_brightness_coefficient = np.random.uniform(-0.35, 1)+0.5 ## generates value between 0.15 and 1.5    
        random_brightness_coefficient = brightness
        
        image_HLS[:,:,1] = image_HLS[:,:,1]*random_brightness_coefficient ## scale pixel values up or down for channel 1(Lightness)    
        image_HLS[:,:,1][image_HLS[:,:,1]>255]  = 255 ##Sets all values above 255 to 255    
        image_HLS = np.array(image_HLS, dtype = np.uint8)
        image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB) ## Conversion to RGB  
        return Image(image_RGB)
    
    def addSunset(self, redMultiplier):
        image_RGB = self.image.copy()
        image_RGB[:,:,2] = image_RGB[:,:,2] * 1.2
        image_RGB[:,:,2][image_RGB[:,:,2]>255] = 255
        return Image(image_RGB)
    
    def addRain(self, slant, rainIntensity):
        
        # slant ~ [-10, 10]
        # rainIntensity ~ [1500, 3000]
        
        image = self.image.copy()
        imshape = image.shape
        
        # slant_extreme=10    
        # slant= np.random.randint(-slant_extreme,slant_extreme)     
        
        drop_length=20    
        drop_width=2    
        drop_color=(200,200,200) ## a shade of gray    
        rain_drops = generate_random_lines(imshape,
                                           slant,
                                           drop_length,
                                           rainIntensity)        
        for rain_drop in rain_drops:        
            cv2.line(image,(rain_drop[0],
                            rain_drop[1]),
                    (rain_drop[0]+slant,rain_drop[1]+drop_length),
                    drop_color,drop_width)
        image= cv2.blur(image,(7,7)) ## rainy view are blurry        
        brightness_coefficient = 0.7 ## rainy days are usually shady     
        image_HLS = cv2.cvtColor(image,cv2.COLOR_RGB2HLS) ## Conversion to HLS    
        image_HLS[:,:,1] = image_HLS[:,:,1]*brightness_coefficient ## scale pixel values down for channel 1(Lightness)    
        image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB) ## Conversion to RGB    
        return Image(image_RGB)
        
    def randomise(self, 
                  hflip=True,
                  color=True,
                  brightness=(0.3, 1.5),
                  sunset=(1.2, 1.4),
                  
                  rain=False,
                  rainSlant=(-10, 10),
                  rainIntensity=(1000,3000)):
        
        image = self.hFlip() if hflip else self
        image = image.addRandomColor() if color else image
        
        brightness = np.random.uniform(*brightness) if brightness else brightness
        image = image.addBrightness(brightness) if brightness else image
        
        redMultiplier = np.random.uniform(*sunset) if sunset else sunset
        image = image.addSunset(redMultiplier) if sunset else image
        
        image = image.addRain(slant = np.random.randint(*rainSlant),
                              rainIntensity = np.random.randint(*rainIntensity)
                              ) if rain else image
        
        return image
    
    @property
    def shape(self):
        return self.image.shape
    
    def imwrite(self, directory):
        cv2.imwrite(directory, self.image)
        
    def scale(self, width=1, height=1):
        # width, height
        dim = (int(self.shape[1]*width), int(self.shape[0]*height))
        # resize image
        resized = cv2.resize(self.image, dim, interpolation = cv2.INTER_AREA)
        return Image(resized)
    
    def addRandomObjects(self, objBank, 
                         objNumber=5, 
                         labelTxt=None,
                         Xrange = (0,1),
                         Yrange = (0.2, 0.5),
                         mixup_ratio=0.3,
                         smoothBorder = False,
                         color = False):
        
        if isinstance(objNumber, tuple):
            objNumber = randint(*objNumber)
            
        sampleImage = self
        newLines = []
        types = []
        for i in range(objNumber):
            ship, imageType = objBank.newObject(hflip = uniform(0,1)>0.5,
                                     color=color)
            types.append(imageType)
            # if the image is too small, enlarge it by a bit
            if ship.shape[0] < 0.1*self.shape[0] or ship.shape[1] < 0.1*self.shape[1]:
                factor = uniform(3,5)
                ship = ship.scale(width=factor, height=factor)
            
                # resize the shape a bit
            if imageType == 0:
                # vessel
                ship = ship.scale(width=uniform(0.8,1.2), height=uniform(0.8,1.2))
            else:
                ship = ship.scale(width=uniform(0.2,0.4), height=uniform(0.2,0.4))
            
            sampleImage, coordinates = sampleImage.mixupWith(ship, 
                                 X = uniform(*Xrange),
                                 Y = uniform(*Yrange),
                                 mixup_ratio=mixup_ratio,
                                 smoothBorder = smoothBorder)
            # print(coordinates)
            newLines.append(coordinates)
        newLines = np.array(newLines).astype(float)
        # print(newLines)
        # print(sampleImage.shape)
        newLines = xyxy2xywhn(x=newLines, w=sampleImage.shape[1],h=sampleImage.shape[0])
        # print(newLines)
        # print(newLines)
        newLines = pd.DataFrame(newLines)
        newLines.insert(0, -1, pd.Series(types))
        newLines.columns = list(range(len(newLines.columns)))
        labelTxt = labelTxt.append(newLines, ignore_index=True)
        # need to edit the labelTxt
        return sampleImage, labelTxt

class RandomObjectBank():
    
    def __init__(self, 
                 vesselbank, kayatBank, 
                 vesselProb = 0.5, 
                 # kayatProb = 1 - vesselProb, 
                 flipProb = 0.5,
                 colorProb = 0.5
                 ):
        assert  0<=vesselProb<=1, 'Probability out of bound'
        
        self.kayatBank = list(i for i in Path(kayatBank).iterdir() if i.suffix == '.jpg')
        self.kayatProb = 1-vesselProb
        
        self.vesselbank = list(i for i in Path(vesselbank).iterdir() if i.suffix == '.jpg')
        self.vesselProb = vesselProb
        
    def newObject(self, hflip, color):
        makeVessel = uniform(0,1) <= self.vesselProb
        if makeVessel:
            image = self.getFromBank(self.vesselbank)
            imageType = 0
        else:
            image = self.getFromBank(self.kayatBank)
            imageType = 1
            
        return image.randomise(hflip=hflip,
                               color=color,
                               brightness=False,
                               sunset=False,
                               rain=False), imageType
    
    def getFromBank(self, bank):
        return Image(choice(bank))

    # sampleMain = "../Data/Vessel_Kayak_Count/images/images/train/GOV1_ExternDisk0_ch2_20200229090000_20200229100000_7740.jpg"
    # vesselPic = '../Data/Vessel_Kayak_Count/images/images/train/labelled/Cropped/GOV1_ExternDisk0_ch2_20200229090000_20200229100000_7704_0_0_.jpg'
    # sampleMain = './Imagenew_006.jpg'
    # sampleImage = Image(sampleMain)
    
    # imagesDir = Path("../Data/fullyAugmented/waterbodies")
    # labelsDir = Path("../Data/fullyAugmented/waterbodies")
    
    # imagesDir = Path('./')
    # labelsDir = Path('./')

def mutate(imageDir_trainTxt_lock, repetition,
        labelsDir,
        imageTargetDir, labelTargetDir,
        objBank, objNumberRange, Xrange, Yrange, 
        mixup_ratio, smoothBorder, color,
        sunsetProb, rainProb,
        brightnessRange):
    
    imageDir, trainTxt, lock  = imageDir_trainTxt_lock
    
    labelDir = labelsDir/(imageDir.stem+'.txt')
    if imageDir.suffix !='.jpg':
        # not a image
        # label not found
        # print(labelDir)
        return
    
    for i in range(repetition):
        image = Image(imageDir)
        
        if not labelDir.is_file():
            # label not found
            labelTxt = pd.DataFrame(columns=list(range(5)))
        else:
            labelTxt = pd.read_csv(labelDir, sep=' ', header=None)
        # a.append(pd.Series([0,0,0,0,0]), ignore_index=True)
        before = labelTxt.shape
        image, labelTxt = image.addRandomObjects(labelTxt=labelTxt,
                                                objBank = objBank, 
                                                objNumber = randint(*objNumberRange),
                                                Xrange = Xrange,
                                                Yrange = Yrange,
                                                mixup_ratio=mixup_ratio,
                                                smoothBorder = smoothBorder,
                                                color = color)
        
        sunset = int(uniform(0,1) < sunsetProb)
        sunset = [False, (1.1, 1.4)][sunset]
        
        rain = uniform(0,1) < rainProb
        rainSlant=(-10, 10)
        rainIntensity=(1000,3000)
                
        image = image.randomise(hflip=False, color=False,
                                brightness=brightnessRange,
                                sunset=sunset,
                                rain=rain,
                                rainSlant=rainSlant,
                                rainIntensity=rainIntensity)
        # print(before, labelTxt.shape)
        image.imwrite(str(imageTargetDir/(imageDir.stem + f'_{i}_.jpg')))
        labelTxt.to_csv(str(labelTargetDir/(labelDir.stem + f'_{i}_.txt')), 
                        sep=' ', header=None, index=False)
        lock.acquire()
        trainTxt.append('./images/train/'+imageDir.stem+f'_{i}_.jpg')
        lock.release()



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-imagesDir', type=str, default="../Data/Vessel_Kayak_Count/images/images/train")
    parser.add_argument('-labelsDir', type=str, default="../Data/Vessel_Kayak_Count/images/labels/train")
    parser.add_argument('-targetDir', type=str, default="../Data/agumentedData_oriMixNew_smoothing_brightness_rain")
    parser.add_argument('-vesselbank', type=str, default='../agumentedDataBank/vessels_oriMixNew')
    parser.add_argument('-kayatBank', type=str, default='../agumentedDataBank/kayats_oriMixNew')
    
    args = parser.parse_args()

    imagesDir = Path(args.imagesDir)
    labelsDir = Path(args.labelsDir)
    targetDir = Path(args.targetDir)
    vesselbank= Path(args.vesselbank)
    kayatBank= Path(args.kayatBank)

    imageTargetDir = targetDir/'images'/'train'
    labelTargetDir = targetDir/'labels'/'train'
    imageTargetDir.mkdir(exist_ok=True, parents=True)
    labelTargetDir.mkdir(exist_ok=True, parents=True)

    objBank = RandomObjectBank(vesselbank=str(vesselbank),
                                kayatBank=str(kayatBank),
                                vesselProb=0.5)
    
    manager = mp.Manager()
    lock = manager.Lock()
    trainTxt = manager.list()
    
    # mp.cpu_count()
    pool = mp.Pool(processes = mp.cpu_count())
    
    mutate_partial = partial(mutate, #(imageDir, trainTxt, lock), 
                            repetition = 50,
                            labelsDir = labelsDir,
                            imageTargetDir = imageTargetDir, 
                            labelTargetDir = labelTargetDir,
                            objBank = objBank, 
                            objNumberRange = (2,6), Xrange = (0,1), Yrange = (0.3, 0.5), 
                            mixup_ratio = 0.2, smoothBorder = True, color = False,
                            sunsetProb = 0, rainProb = 0.2,
                            brightnessRange = (0.3, 1.5))
    
    imagesCount = len(list(imagesDir.iterdir()))
    params = zip(imagesDir.iterdir(), 
                 repeat(trainTxt),
                 repeat(lock))
    
    for _ in tqdm(pool.imap(mutate_partial, params),
                  desc='Progress',
                  total=imagesCount):
        continue
    
    trainTxt = "\n".join(trainTxt)
    with open(str(targetDir/'train.txt'), 'w') as f:
        f.write(trainTxt)
        
# python3 newDataGeneratorMultiproc.py -imagesDir ../Data/Vessel_Kayak_Count/images/images/train -labelsDir ../Data/Vessel_Kayak_Count/images/labels/train -targetDir ../Data/agumentedData_oriMixNew_smoothing_brightness_rain_argparseTest -vesselbank ../agumentedDataBank/vessels_oriMixNew -kayatBank ../agumentedDataBank/kayats_oriMixNew
