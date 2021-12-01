# iteratively look through a folder of pictures
# remove pictures that may be too similar
# this is to filter the cropped ships/kayak
# that is generated from `drawBoxes`

import cv2
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path



def imageSimilarity(image1, image2):
    try:
        sim = cv2.matchTemplate(image1,
                                image2,
                                cv2.TM_CCOEFF_NORMED)
        return max(sim)[0]
    except:
        
        height = max(image1.shape[0], image2.shape[0])
        width = max(image1.shape[1], image2.shape[1])
        mix_img = np.zeros((height, width, 3),dtype=np.float32)
        mix_img[:image1.shape[0], :image1.shape[1], :] = image1
        image1 = mix_img
        
        mix_img = np.zeros((height, width, 3),dtype=np.float32)
        mix_img[:image2.shape[0], :image2.shape[1], :] = image2
        image2 = mix_img
        
        sim = cv2.matchTemplate(image1,
                                image2,
                                cv2.TM_CCOEFF_NORMED)
        # print((i,j, max(sim)[0]))
        return max(sim)[0]
    

src = Path('../Data/shipCrops/kayats/')
kayats = list(src.iterdir())
kayatTrash = src/'trash'

pics = [
    'GOV1_ExternDisk0_ch2_20200229100000_20200229110000_10080_2_1_.jpg',
    'GOV1_ExternDisk0_ch2_20200229100000_20200229110000_10116_2_1_.jpg',
    
    'frame25742_0_1_.jpg',
    'frame25806_0_1_.jpg',
    'frame25832_0_1_.jpg',
]

def removeSimilarImages(path, similarityThreshold=0.1):
    src = Path(path)
    images = list(image for image in src.iterdir() if image.suffix == '.jpg' )
    imagesTrash = src/'trash'
    
    i = 0
    progress = tqdm()
    count = 0
    while i < len(images)-1:
        # trashPhotos = []
        imageI = cv2.imread(str(images[i]))
        
        for j in range(len(images)-1, i, -1):
            similarity = imageSimilarity(
                imageI,
                cv2.imread(str(images[j]))
            )
            # trashPhotos.append(j) if similarity >= similarityThreshold else None
            if similarity >= similarityThreshold:
                count += 1
                images[j].rename(imagesTrash/images[j].name)
                del images[j]

        i += 1
        progress.update(1)
        progress.set_description(f'{count} images moved')

if __name__ == '__main__':
    src = Path('../Data/shipCrops/kayats/')
    removeSimilarImages(src, 0.05)

    src = Path('../Data/shipCrops/vessels/')
    removeSimilarImages(src, 0.1)


            
             
        