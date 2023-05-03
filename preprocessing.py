import os
import json
import h5py
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from shapely.geometry import box

H5_DIR = 'h5_data'
TMA_DATA_DIR = 'data/TMA'
PCAM_DATA_DIR = 'data/pcam'
PATCHES_DIR_TMA = 'data/TMA/train'
PCAM_DIR = 'data/pcam/train'
OPENSLIDE_PATH = 'C:/OpenSlide/openslide-win64-20221217/bin'
FILE_FORMAT = 'camelyonpatch_level_2_split_{}_{}.h5'
ANNOTATIONS_PATH = 'annotations.csv'
DOWNSAMPLE_FACTOR = 10
SATURATION_THRESHOLD = 0.35
BRIGHTNESS_THRESHOLD = 0.7

with os.add_dll_directory(OPENSLIDE_PATH):
    import openslide

def h5_to_jpeg():
    try:
        os.makedirs(PCAM_DATA_DIR)
    except:
        print('Data already prepared')
        return
    
    for stage in ['train', 'test', 'valid']:
        inner_path = os.path.join(PCAM_DATA_DIR, stage)
        os.makedirs(inner_path)
        
        images = h5py.File(os.path.join(H5_DIR, FILE_FORMAT.format(stage, 'x')), 'r')
        labels = h5py.File(os.path.join(H5_DIR, FILE_FORMAT.format(stage, 'y')), 'r')
        x_key = list(images.keys())[0]
        y_key = list(labels.keys())[0]
        labels_dict = {}
        
        for index, image in enumerate(images[x_key]):
            img_name = 'img_' + str(index) + '.jpeg'
            cv2.imwrite(os.path.join(inner_path, img_name), image)
            labels_dict[index] = (img_name, labels[y_key][index].flatten().tolist()[0])
            
        with open(os.path.join(inner_path, 'labels.json'), 'w') as f:
            json.dump(labels_dict, f, indent=4)

def check_thresholds(patch_hsv):
    hue_channel = patch_hsv[:, :, 0]
    saturation_channel = patch_hsv[:, :, 1]
    brightness_channel = patch_hsv[:, :, 2]
    patch_hue = np.mean(hue_channel / 255)
    patch_saturation = np.mean(saturation_channel / 255)
    patch_brightness = np.mean(brightness_channel / 255)
    if (patch_saturation > 0.4 and patch_brightness > 0.5 and patch_brightness < 0.7 and patch_hue > 0.5 and patch_hue < 0.6) or \
        (patch_saturation > 0.3 and patch_brightness > 0.7 and patch_brightness < 0.8 and patch_hue > 0.5 and patch_hue < 0.6):
        return True
    return False
    
def filter_pcam():
    save_dir = 'data/pcam/trainHE'
    waste = 'data/pcam/waste'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        print('Data already filtered')
        return
    if not os.path.exists(waste):
        os.makedirs(waste)
    labels_file = os.listdir(PCAM_DIR)[-1]
    f = open(os.path.join(PCAM_DIR, labels_file))
    labels = json.load(f)
    new_labels = {}
    index = 0
    
    for _, file in enumerate(list(labels.values())):
        image = cv2.imread(os.path.join(PCAM_DIR, file[0]))
        patch_hsv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)
        patch_hsv = cv2.GaussianBlur(patch_hsv, (5, 5), 0)

        if check_thresholds(patch_hsv):
            patch_name = 'img_' + str(index) + '.jpeg'
            cv2.imwrite(os.path.join(save_dir, patch_name), image)
            new_labels[index] = (patch_name, file[1])
            index += 1
        else:
            cv2.imwrite(os.path.join(waste, file[0]), image)

    with open(os.path.join(save_dir, 'labels.json'), 'w') as f:
        json.dump(new_labels, f, indent=4)

def get_positive_patches_TMA(labels):
    csv_path = os.path.join(TMA_DATA_DIR, ANNOTATIONS_PATH)
    df = pd.read_csv(csv_path)
    
    if not os.path.exists(PATCHES_DIR_TMA):
        os.makedirs(PATCHES_DIR_TMA)
    
    patch_size = (96, 96)
    index = 0
    
    for _, row in df.iterrows():
        if row['stain'] != 'HE':
            continue
        slide_path = os.path.join(TMA_DATA_DIR, row['stain'], row['tma_id'])
        slide = openslide.open_slide(slide_path)
        level = slide.get_best_level_for_downsample(DOWNSAMPLE_FACTOR)
        scale_factor = 1 / int(slide.level_downsamples[level])
        roi_start = (row['xs'], row['ys'])
        roi_size = (int((row['xe'] - row['xs']) * scale_factor), int((row['ye'] - row['ys']) * scale_factor))
        roi = slide.read_region(roi_start, level, roi_size)

        for y in range(0, roi.size[1] - patch_size[1], patch_size[1]):
            for x in range(0, roi.size[0] - patch_size[0], patch_size[0]):
                patch_rgb = roi.crop((x, y, x + patch_size[0], y + patch_size[1]))
                patch_rgb = cv2.cvtColor(np.array(patch_rgb), cv2.COLOR_RGBA2BGR)
                patch_hsv = cv2.cvtColor(patch_rgb, cv2.COLOR_BGR2HSV)
                patch_hsv = cv2.GaussianBlur(patch_hsv, (5, 5), 0)
                
                saturation_channel = patch_hsv[:, :, 1]
                brightness_channel = patch_hsv[:, :, 2]
                patch_saturation = np.mean(saturation_channel / 255)
                patch_brightness = np.mean(brightness_channel / 255)
                
                if patch_saturation > SATURATION_THRESHOLD and patch_brightness < BRIGHTNESS_THRESHOLD:
                # if check_thresholds(patch_hsv):
                    patch_name = 'img_' + str(index) + '.jpeg'
                    cv2.imwrite(os.path.join(PATCHES_DIR_TMA, patch_name), patch_rgb)
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

def get_negative_patches_TMA(labels):
    csv_path = os.path.join(TMA_DATA_DIR, ANNOTATIONS_PATH)
    annotations = parse_annotations(csv_path)

    if not os.path.exists(PATCHES_DIR_TMA):
        os.makedirs(PATCHES_DIR_TMA)
        
    patch_size = (96, 96)
    index = len(labels)
    
    for tma_id, regions in annotations.items():
        slide_path = os.path.join(TMA_DATA_DIR, 'HE', tma_id)
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
                patch_rgb = roi.crop((x, y, x + patch_size[0], y + patch_size[1]))
                patch_rgb = cv2.cvtColor(np.asarray(patch_rgb), cv2.COLOR_RGBA2BGR)
                patch_hsv = cv2.cvtColor(patch_rgb, cv2.COLOR_BGR2HSV)
                patch_hsv = cv2.GaussianBlur(patch_hsv, (5, 5), 0)

                saturation_channel = patch_hsv[:, :, 1]
                brightness_channel = patch_hsv[:, :, 2]

                patch_saturation = np.mean(saturation_channel / 255)
                patch_brightness = np.mean(brightness_channel / 255)
                
                if patch_saturation > SATURATION_THRESHOLD and patch_brightness < BRIGHTNESS_THRESHOLD:
                # if check_thresholds(patch_hsv):
                    patch_name = 'img_' + str(index) + '.jpeg'
                    cv2.imwrite(os.path.join(PATCHES_DIR_TMA, patch_name), patch_rgb)
                    labels[index] = (patch_name, 0)
                    index += 1
    
    with open(os.path.join(PATCHES_DIR_TMA, 'labels.json'), 'w') as f:
        json.dump(labels, f, indent=4)
    
if __name__ == '__main__':
    labels = {}
    h5_to_jpeg()
    filter_pcam()
    get_positive_patches_TMA(labels)
    get_negative_patches_TMA(labels)