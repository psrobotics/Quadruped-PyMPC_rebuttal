"""
    File: composite_image.py
    Author: renyunfan
    Email: renyf@connect.hku.hk
    Description: [ A python script to create a composite image from a video.]
    All Rights Reserved 2023
"""

import cv2
import numpy as np
from enum import Enum
import argparse
from pathlib import Path
from natsort import natsorted   # pip install natsort

class CompositeMode(Enum):
    MAX_VARIATION = 0
    MIN_VALUE = 1
    MAX_VALUE = 2


class CompositeImage:

    def __init__(self, mode, video_path, start_t = 0, end_t = 999, skip_frame = 1):
        self.video_path = video_path
        self.skip_frame = skip_frame
        self.start_t = start_t
        self.end_t = end_t
        self.mode = mode

    def max_variation_update(self, image, alpha_k):
        delta_img = image - self.ave_img
        image_norm = np.linalg.norm(image, axis=2)
        delta_norm = image_norm - self.ave_img_norm
        abs_delta_norm = np.abs(delta_norm)
        delta_mask = abs_delta_norm > self.abs_diff_norm
        diff_mask = abs_delta_norm <= self.abs_diff_norm
        delta_mask = np.stack((delta_mask.T, delta_mask.T, delta_mask.T)).T.astype(np.float32)
        diff_mask = np.stack((diff_mask.T, diff_mask.T, diff_mask.T)).T.astype(np.float32)
        self.diff_img = self.diff_img * diff_mask  + delta_img * delta_mask*alpha_k
        self. diff_norm = np.linalg.norm(self.diff_img, axis=2)
        self.abs_diff_norm = np.abs(self.diff_norm)

    def min_value_update(self, image):
        image_norm = np.linalg.norm(image, axis=2)
        cur_min_image = self.diff_img + self.ave_img
        cur_min_image_norm = np.linalg.norm(cur_min_image,axis=2)
        delta_mask = cur_min_image_norm > image_norm
        min_mask = cur_min_image_norm <= image_norm
        delta_mask = np.stack((delta_mask.T, delta_mask.T, delta_mask.T)).T.astype(np.float32)
        min_mask = np.stack((min_mask.T, min_mask.T, min_mask.T)).T.astype(np.float32)
        new_min_img = image * delta_mask + min_mask * cur_min_image
        self.diff_img = new_min_img - self.ave_img

    import numpy as np

    def max_value_update(self, image):
        image_norm = np.linalg.norm(image, axis=2)
        cur_min_image = self.diff_img + self.ave_img
        cur_min_image_norm = np.linalg.norm(cur_min_image, axis=2)
        delta_mask = cur_min_image_norm < image_norm
        min_mask = cur_min_image_norm >= image_norm
        delta_mask = np.stack((delta_mask, delta_mask, delta_mask), axis=2).astype(np.float32)
        min_mask = np.stack((min_mask, min_mask, min_mask), axis=2).astype(np.float32)
        new_min_img = image * delta_mask + min_mask * cur_min_image
        self.diff_img = new_min_img - self.ave_img

    def extract_frames(self):
        base = Path(self.video_path)
        # find all rollout render dirs: e.g. 1/render, 2/render, ...
        rollout_dirs = sorted([d / "render" 
                            for d in base.iterdir() 
                            if d.is_dir() and (d / "render").is_dir()],
                            key=lambda p: int(p.parent.name))

        self.img_paths = []
        self.imgs = []
        #for rollout_dir in rollout_dirs:
        rollout_dir = rollout_dirs[0]
        # list and sort all image file paths
        rollout_paths = natsorted([
            p for p in rollout_dir.iterdir() 
            if p.suffix.lower() in (".png", ".jpg", ".jpeg")
        ])
        for img_path in rollout_paths:
            self.img_paths.append(img_path)
            # load image once here
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"Failed to read image: {img_path}")
            self.imgs.append(img)

        return self.imgs  # optionally return, too

    def merge_images(self):
        image_files = self.extract_frames()
        if(image_files == None or len(image_files) < 1):
            print("Error: no image extracted, input video path at: ", self.video_path)
            exit(1)
        first_image = image_files[0]
        height, width, _ = first_image.shape

        # 遍历每张图片，将像素值取最大值并合成到空画布上
        sum_image = np.zeros((height, width, 3), dtype=np.float32)
        img_num = len(image_files)

        for image_file in image_files:
            image = image_file.astype(np.float32)
            sum_image += image

        self.ave_img = sum_image / img_num
        self.first_img = image_files[0].astype(np.float32)

        self.ave_img_norm = np.linalg.norm(self.ave_img, axis=2)
        self.diff_norm = np.zeros((height, width), dtype=np.float32)
        self.abs_diff_norm = np.zeros((height, width), dtype=np.float32)
        self.diff_img = np.zeros((height, width, 3), dtype=np.float32)


        cnt = 0
        for image_file in image_files:
            cnt = cnt + 1
            alpha_k = 0.2+0.8*cnt/img_num #1.0-0.5*cnt/img_num
            print(alpha_k)
            print("Processing ", cnt, " / ", img_num)
            image = image_file.astype(np.float32)
            # get sub image
            if (cnt % self.skip_frame == 0):
                if(self.mode == CompositeMode.MAX_VARIATION):
                    self.max_variation_update(image, alpha_k)
                elif(self.mode == CompositeMode.MIN_VALUE):
                    self.min_value_update(image)
                elif(self.mode == CompositeMode.MAX_VALUE):
                    self.max_value_update(image)
            #cv2.imwrite('seq_f/'+'seq_i_'+str(cnt)+'.jpg', image)

        merged_image = self.ave_img + self.diff_img
        merged_image = merged_image.astype(np.uint8)
        #cv2.imwrite('diff.jpg', self.diff_img.astype(np.uint8))
        return merged_image

# 读取命令行参数
path = "test_save/nonfriction_lt"
mode = "VAR"
start_t = 0
end_t = 99999999
skip_frame = 3

print(" -- Load Param: video path", path)
print(" -- Load Param: mode", mode)
print(" -- Load Param: start_t", start_t)
print(" -- Load Param: end_t", end_t)
print(" -- Load Param: skip_frame", skip_frame)

if(mode == 'MAX'):
    mode = CompositeMode.MAX_VALUE
elif(mode == 'MIN'):
    mode = CompositeMode.MIN_VALUE
elif(mode == 'VAR'):
    mode = CompositeMode.MAX_VARIATION


merger = CompositeImage(mode, path,start_t,end_t, skip_frame)
merged_image = merger.merge_images()
cv2.imwrite('nonfriction_lt.png', merged_image)
