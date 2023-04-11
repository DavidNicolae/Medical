import os
import json
import pandas as pd
import numpy as np
import cv2
import shapely
from shapely.geometry import box
from PIL import Image, ImageFilter

DATA_DIR = 'data/TMA'
POSITIVE_PATCHES_DIR = 'data/TMA/positive'
NEGATIVE_PATCHES_DIR = 'data/TMA/negative'
DISCARDED_DIR = 'data/TMA/discarded'
OPENSLIDE_PATH = 'C:/OpenSlide/openslide-win64-20221217/bin'
ANNOTATIONS_PATH = 'annotations.csv'
DOWNSAMPLE_FACTOR = 10
SATURATION_THRESHOLD = 0.07
BRIGHTNESS_THRESHOLD = 0.5

with os.add_dll_directory(OPENSLIDE_PATH):
    import openslide

def extract_positive_patches():
    csv_path = os.path.join(DATA_DIR, ANNOTATIONS_PATH)
    df = pd.read_csv(csv_path)
    
    if not os.path.exists(POSITIVE_PATCHES_DIR):
        os.makedirs(POSITIVE_PATCHES_DIR)
    if not os.path.exists(DISCARDED_DIR):
        os.makedirs(DISCARDED_DIR)
    
    patch_size = (96, 96)
    labels_dict = {}
    index = 0
    discard_index = 0
    
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
        
        # print(roi.size)
        # whole_slide = slide.read_region((0, 0), level, slide.level_dimensions[level])
        # roi = roi.resize((int(roi_size[0] / downsample_factor), int(roi_size[1] / downsample_factor)))

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
                    patch_rgb.save(os.path.join(POSITIVE_PATCHES_DIR, patch_name), 'jpeg')
                    labels_dict[index] = (patch_name, 1)
                    index += 1
                else:
                    discard_index += 1
    print(discard_index)
                    
    # with open(os.path.join(DATA_DIR, 'labels.json'), 'w') as f:
    #     json.dump(labels_dict, f, indent=4)
        
    # image = slide.read_region((0, 0), 0, (width, height))
    # image = Image.frombytes('RGB', (width, height), image)
    # image.show()

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

def extract_negative_patches():
    csv_path = os.path.join(DATA_DIR, ANNOTATIONS_PATH)
    annotations = parse_annotations(csv_path)

    if not os.path.exists(NEGATIVE_PATCHES_DIR):
        os.makedirs(NEGATIVE_PATCHES_DIR)
    if not os.path.exists(DISCARDED_DIR):
        os.makedirs(DISCARDED_DIR)
        
    patch_size = (96, 96)
    labels_dict = {}
    index = 0
    discard_index = 0
    
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
                        discard_index += 1
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
                    patch_rgb.save(os.path.join(NEGATIVE_PATCHES_DIR, patch_name), 'jpeg')
                    labels_dict[index] = (patch_name, 1)
                    index += 1
                else:
                    discard_index += 1
    print(discard_index)

if __name__ == '__main__':
    extract_positive_patches()
    extract_negative_patches()