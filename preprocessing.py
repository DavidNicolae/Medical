import os
import json
import pandas as pd
import numpy as np
import cv2
from shapely.geometry import box

DATA_DIR = 'data/TMA'
PATCHES_DIR = 'data/TMA/train'
OPENSLIDE_PATH = 'C:/OpenSlide/openslide-win64-20221217/bin'
ANNOTATIONS_PATH = 'annotations.csv'
DOWNSAMPLE_FACTOR = 10
SATURATION_THRESHOLD = 0.07
BRIGHTNESS_THRESHOLD = 0.5

with os.add_dll_directory(OPENSLIDE_PATH):
    import openslide

def extract_positive_patches(labels):
    csv_path = os.path.join(DATA_DIR, ANNOTATIONS_PATH)
    df = pd.read_csv(csv_path)
    
    if not os.path.exists(PATCHES_DIR):
        os.makedirs(PATCHES_DIR)
    
    patch_size = (96, 96)
    index = 0
    
    for _, row in df.iterrows():
        if row['stain'] != 'HE':
            continue
        slide_path = os.path.join(DATA_DIR, row['stain'], row['tma_id'])
        slide = openslide.open_slide(slide_path)
        level = slide.get_best_level_for_downsample(DOWNSAMPLE_FACTOR)
        scale_factor = 1 / int(slide.level_downsamples[level])
        roi_start = (row['xs'], row['ys'])
        roi_size = (int((row['xe'] - row['xs']) * scale_factor), int((row['ye'] - row['ys']) * scale_factor))
        roi = slide.read_region(roi_start, level, roi_size)

        for y in range(0, roi.size[1] - patch_size[1], patch_size[1]):
            for x in range(0, roi.size[0] - patch_size[0], patch_size[0]):
                patch_rgb = roi.crop((x, y, x + patch_size[0], y + patch_size[1])).convert('RGB')
                patch_hsv = cv2.cvtColor(np.array(patch_rgb), cv2.COLOR_RGB2HSV)
                patch_hsv = cv2.GaussianBlur(patch_hsv, (5, 5), 0)
                saturation_channel = patch_hsv[:, :, 1]
                brightness_channel = patch_hsv[:, :, 2]
                
                patch_saturation = np.mean(saturation_channel / 255)
                patch_brightness = np.mean(brightness_channel / 255)
                
                if patch_saturation > SATURATION_THRESHOLD and patch_brightness > BRIGHTNESS_THRESHOLD:
                    patch_name = 'img_' + str(index) + '.jpeg'
                    patch_rgb.save(os.path.join(PATCHES_DIR, patch_name), 'jpeg')
                    labels[index] = (patch_name, 1)
                    index += 1

def parse_annotations(csv_path):
    annotations = {}
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        if row['stain'] == 'HE':
            roi = (row['xs'], row['ys'], row['xe'], row['ye'])
            if row['tma_id'] in annotations:
                annotations[row['tma_id']].append(roi)
            else:
                annotations[row['tma_id']] = [roi]
    return annotations

def extract_negative_patches(labels):
    csv_path = os.path.join(DATA_DIR, ANNOTATIONS_PATH)
    annotations = parse_annotations(csv_path)

    if not os.path.exists(PATCHES_DIR):
        os.makedirs(PATCHES_DIR)
        
    patch_size = (96, 96)
    index = len(labels)
    
    for tma_id, regions in annotations.items():
        slide_path = os.path.join(DATA_DIR, 'HE', tma_id)
        slide = openslide.open_slide(slide_path)
        level = slide.get_best_level_for_downsample(DOWNSAMPLE_FACTOR)
        scale_factor = 1 / int(slide.level_downsamples[level])
        roi = slide.read_region((0, 0), level, slide.level_dimensions[level])
        
        for y in range(0, roi.size[1] - patch_size[1], patch_size[1]):
            for x in range(0, roi.size[0] - patch_size[0], patch_size[0]):
                patch = box(x, y, x + patch_size[0], y + patch_size[1])
                intersection = False
                for region in regions:
                    if patch.intersects(box(region[0] * scale_factor, region[1] * scale_factor,
                                            region[2] * scale_factor, region[3] * scale_factor)):
                        intersection = True
                        break
                if intersection == True:
                    continue
                patch_rgb = roi.crop((x, y, x + patch_size[0], y + patch_size[1])).convert('RGB')
                patch_hsv = cv2.cvtColor(np.array(patch_rgb), cv2.COLOR_RGB2HSV)
                patch_hsv = cv2.GaussianBlur(patch_hsv, (5, 5), 0)
                saturation_channel = patch_hsv[:, :, 1]
                brightness_channel = patch_hsv[:, :, 2]
                
                patch_saturation = np.mean(saturation_channel / 255)
                patch_brightness = np.mean(brightness_channel / 255)
                
                if patch_saturation > SATURATION_THRESHOLD and patch_brightness > BRIGHTNESS_THRESHOLD:
                    patch_name = 'img_' + str(index) + '.jpeg'
                    patch_rgb.save(os.path.join(PATCHES_DIR, patch_name), 'jpeg')
                    labels[index] = (patch_name, 0)
                    index += 1
    
    with open(os.path.join(PATCHES_DIR, 'labels.json'), 'w') as f:
        json.dump(labels, f, indent=4)

if __name__ == '__main__':
    labels = {}
    extract_positive_patches(labels)
    extract_negative_patches(labels)
    print(len(labels))