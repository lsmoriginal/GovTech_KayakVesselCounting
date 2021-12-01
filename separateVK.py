# splitting the images into kayak and vessels

import os

targetPath = '../Data/shipCrops'
kayat = '../Data/kayats'
vessels = '../Data/vessels'

for image in os.listdir(targetPath):
    if image.endswith('0_.jpg'):
        os.rename(os.path.join(targetPath, image),
                  os.path.join(vessels, image))
    else:
        os.rename(os.path.join(targetPath, image),
                  os.path.join(kayat, image))