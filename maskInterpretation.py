import cv2
from newDataGenerator import Image

sampleMaskImage = '../data/mask/CustomVessel_5.jpg'

images = cv2.imread(sampleMaskImage)


class Mask(Image):
    
    def __init__(self, image):
        super().__init__(image)
    
    @property
    def mask(self, 
             maskValue=0, 
             objectValue=1,
             smoothing=0):
        '''
        maskValue: value in the picture that represent a block, 
                   default to zero -> Black
        returns a N*M np.ndarray representing the mask
        '''
        image = self.image.copy()
        
        # true bool indicating object target
        maskBool = ((image == maskValue).sum(axis=2) != 3)
        
        if not smoothing:
            return maskBool
        
        
    
    
        
        