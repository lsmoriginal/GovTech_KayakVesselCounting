# take screenshots from the video at specified intervals

import cv2
import argparse
import os
import datetime
from tqdm import tqdm

def snapshots(video,
              destination,
              interval):
    
    cap = cv2.VideoCapture(video)
    # assert(cap.isOpened(), "Error opening video stream or file")
    
    videoDir = os.path.dirname(video)
    videoBaseName = os.path.basename(video)
    videoBaseName = videoBaseName[:videoBaseName.find('.')]
    if not destination:
        destination = os.path.join(videoDir, videoBaseName+'_snapshots')
    if not os.path.exists(destination):
        os.makedirs(destination)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count_len = len(str(frame_count))
    finterval = int(fps * interval)
    framePos = 0
    
    mybar = tqdm(total = frame_count, desc = "Progress")
    t_loopStart = datetime.datetime.now()
    barAcc = 0
    while framePos <= frame_count:
        
        cap.set(cv2.CAP_PROP_POS_FRAMES,framePos)
        framePos += finterval
        ret, frame = cap.read()
        if not ret:
            continue
        
        # convert duration format
        durSeconds = framePos/fps
        frameTime = str(datetime.datetime.fromtimestamp(durSeconds) - datetime.datetime.fromtimestamp(0))
        # print(frameTime)
        frameTime = frameTime.replace(':','_')
        frameTime = frameTime.replace('.','_')
        frameTime = frameTime[:11]
        frameName = f'frame_{framePos:_>{str(frame_count_len)}}_'+ frameTime + '.jpg'
        cv2.imwrite(os.path.join(destination, frameName), frame)
        
        
        barAcc += finterval
        if (datetime.datetime.now() - t_loopStart).total_seconds() > 2:
            mybar.update(barAcc)
            barAcc = 0 # reset the values
            t_loopStart = datetime.datetime.now()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video', type=str)
    parser.add_argument('-d', '--destination', 
                        default = '',type=str)
    parser.add_argument('-i', '--interval', 
                        default = 30, type=float,
                        help='time interval between snapshots(seconds)')
    
    # example:
    # python3 videoSnapshot.py -v ../Data/Vessel_Kayak_Count/videos/Team_2/ExternDisk0_ch1_20200305130000_20200305140000.avi
    # python3 videoSnapshot.py -v ../Data/youtubeVideo/test1.mp4 -i 1
    args = parser.parse_args()
    snapshots(**vars(args))
    