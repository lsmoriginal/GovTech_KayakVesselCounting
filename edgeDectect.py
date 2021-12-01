import cv2
import os
import numpy as np 


samplePicsDir = '../Data/edgeDetectionData'
picBases = os.listdir(samplePicsDir)


image = cv2.imread(os.path.join(samplePicsDir, picBases[0]))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# show the original and blurred images
print(image.shape)
print(gray.shape)
blurredRGB = cv2.cvtColor(blurred, cv2.COLOR_GRAY2RGB)
print(blurred.shape)

combined = np.hstack([image, blurredRGB])
# cv2.imshow("Original", image)
# cv2.imshow("Blurred", blurred)
wide = cv2.Canny(blurred, 400, 250)
tight = cv2.Canny(blurred, 240, 250)
mid = cv2.Canny(blurred, 240, 400)
# combined = np.hstack([wide, mid, tight])


cv2.imshow("Window", combined)
cv2.waitKey(0)