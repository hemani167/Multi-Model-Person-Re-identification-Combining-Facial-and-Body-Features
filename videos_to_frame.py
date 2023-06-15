import cv2
import os

# Open the video file
vidcap = cv2.VideoCapture('RESULTS/final_results/outdoor.mp4')

# Create a folder to store the frames
if not os.path.exists('outdoor_frames'):
    os.makedirs('outdoor_frames')

# Read the first frame
success,image = vidcap.read()
count = 0

# Loop through the video and save each frame as an image
while success:
    # Save the frame as an image
    ("frame:",count)
    cv2.imwrite("outdoor_frames/frame%d.jpg" % count, image)     
    success,image = vidcap.read()
    count += 1
