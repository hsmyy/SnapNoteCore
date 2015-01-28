__author__ = 'fc'

import cv2
import os

output = "output/"
files = os.listdir(output)

for file in files:
    img = cv2.imread(output + file)
    