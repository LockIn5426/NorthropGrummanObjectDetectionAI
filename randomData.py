import glob
from PIL import Image
import pillow_avif

import shutil

names = glob.glob('./MicrosoftCOCO.v2-raw.yolov8/train/images/*.jpg')

percent = .01

for n in range(int(len(names) * percent)):
    source = names[n]
    dest = "./newMages/"
    shutil.copy(source, dest)
