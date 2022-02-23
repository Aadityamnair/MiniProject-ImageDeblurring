import cv2
import os
import numpy as np
from tqdm import tqdm

os.makedirs('input/gaussian_blurred', exist_ok=True)

name_list = os.listdir('input/sharp')
for name in tqdm(name_list):
    load_path = os.path.join('input/sharp', name)
    save_path = os.path.join('input/gaussian_blurred', name)
    image = cv2.imread(load_path,cv2.IMREAD_ANYCOLOR )
    blur = cv2.GaussianBlur(image, (31, 31), 0)
    cv2.imwrite(save_path, blur)
print("done")
