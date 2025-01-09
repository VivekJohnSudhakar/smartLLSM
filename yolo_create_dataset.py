#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 11:13:21 2021

@author: jimmytabet
"""

#%% imports
import os,glob,random,datetime
import numpy as np
from scipy import ndimage
import matplotlib
import tifffile as tiff
import nibabel as nib
import math

#%% split Position folders into train and test
'''
path = '/home/nel/NEL-LAB Dropbox/NEL/Datasets/smart_micro/Cellpose_tiles/annotation_results'
all_files = sorted(glob.glob(os.path.join(path,'**','*.npz'), recursive=True))

data_1 = '/home/nel/NEL-LAB Dropbox/NEL/Datasets/smart_micro/Cellpose_tiles/annotation_results/data_1_cellpose'
data_2 = '/home/nel/NEL-LAB Dropbox/NEL/Datasets/smart_micro/Cellpose_tiles/annotation_results/data_2_cellpose'

data_1_folders = os.listdir(data_1)
data_2_folders = os.listdir(data_2)

d1 = [os.path.join(data_1, i) for i in data_1_folders]
d2 = [os.path.join(data_2, i) for i in data_2_folders]

position_folders = d1+d2

# randomize
random.shuffle(position_folders)

train_files = []
test_files = []

i = 0
while len(train_files) <= int(.7*len(all_files)):
    files = glob.glob(os.path.join(position_folders[i], '*.npz'))
    train_files += files
    i += 1

for folder in position_folders[i:]:
    files = glob.glob(os.path.join(folder, '*.npz'))
    test_files += files
'''
#%% set classes
train_files = [r'S:\vivek\cellpose_3D\masked_images_with_img\annotation_results\cellpose_3D\masked_images_with_img_custom_func_test_results\crop_232_5_7_annotated.npz'] #annotation_results\cellpose_3D\masked_images_with_img_results\crop_232_5_7_annotated.npz']
test_files = []
classes = ['anaphase', 'blurry', 'interphase', 'metaphase', 'prometaphase', 'prophase', 'telophase']

#%% loop through train and test - takes ~3.5 min
import time

start = time.time()

bb_size = 70

#base_folder = '/home/nel/Software/yolov5/smart_micro_datasetsv2'
base_folder = r'S:\vivek\cellpose_3D\masked_images_with_img\annotation_results\cellpose_3D' #masked_images_with_img
for mode in ['train']:#, 'test']:
    
    print(mode.upper())

    if mode == 'train':
        files = train_files
    elif mode == 'test':
        files = test_files
    else:
        raise ValueError('not train or test mode!')
            
    count = 0
    for file in files:
        dat = np.load(file, allow_pickle=True)
        print(file)
        raw = dat['raw']
        masks = dat['masks']
        labels = dat['labels']
        print("raw shape: ",raw.shape)
        print("masks shape: ",masks.shape)
        #print("labels: ",labels)
        #exit()
        #raw = raw.transpose((2, 1, 0)) #raw.transpose((1, 2, 0))
        raw_nifti = raw.transpose((2, 1, 0)) #raw.transpose((1, 2, 0))
        #masks = masks.transpose((2, 1, 0)) #masks.transpose((1, 2, 0))
        masks_nifti = masks.transpose((2, 1, 0)) #masks.transpose((1, 2, 0))
        affine = np.eye(4)
        raw_nifti = nib.Nifti1Image(raw_nifti,affine)
        #masks_nifti = nib.Nifti1Image(masks,affine)
        masks_nifti = nib.Nifti1Image(masks_nifti,affine)
        #matplotlib.image.imsave(os.path.join(base_folder, 'images', mode, f'{count}.jpg'), raw, cmap='gray', dpi=300)
        #tiff.imwrite(os.path.join(base_folder, 'images', mode, f'{count}.tif'), raw) 
        #tiff.imwrite(os.path.join(base_folder, 'images', mode, f'{count+1}.tif'),masks) 
        
        nib.save(raw_nifti,os.path.join(base_folder, 'images', mode, f'{count}.nii'))
        nib.save(masks_nifti, os.path.join(base_folder, 'images', 'mask', mode, f'{count}_masks_cellpose.nii'))
        num_masks = masks.max()
        
        with open(os.path.join(base_folder, 'labels', mode, f'{count}.txt'),'w+') as f:#w+
        
            for idx, mask_id in enumerate(range(1,num_masks+1)):
                
                if labels[idx] not in classes:
                    continue
                conditionMet = (masks==mask_id)
                print("mask id: ",mask_id)
                #print(conditionMet.shape)
                #print(conditionMet)
                mask_indices  = np.nonzero(conditionMet)
                #print("z: ",mask_indices[0][-1] - mask_indices[0][0]+1)
                #print("y: ",mask_indices[1][-1] - mask_indices[1][0]+1)
                #print("x: ",mask_indices[2][-1] - mask_indices[2][0]+1)
                #This calculates the depth,widthand height of the cell so the bounding box shape is proportional to the cell size
                cell_depth_z = mask_indices[0][-1] - mask_indices[0][0]+1
                #cell_depth_z2 = mask_indices[0].max() - mask_indices[0].min()+1
                #print("1: ",mask_indices[0].max(),mask_indices[0].min())
                #print(mask_indices[0].min() + (mask_indices[0].max() - mask_indices[0].min())//2)
                cell_height_y = mask_indices[1].max()-mask_indices[1].min()+1#mask_indices[1][-1] - mask_indices[1][0]+1
                cell_width_x = mask_indices[2].max()-mask_indices[2].min()+1#mask_indices[2][-1] - mask_indices[2][0]+1
                zIndex = np.where(conditionMet.any(axis=(1, 2)))
                #print(zIndex)
                #print(masks[zIndex].shape)
                #exit()
                # find center of mass of cell
                center = ndimage.center_of_mass(masks==mask_id)
                #print("mask_id: ",mask_id)
                #print("center : ",(center))
                #center = np.array(center)/raw.shape[0]
                center = np.array(center)
                center[0] = center[0]/raw.shape[0]
                center[1] = center[1]/raw.shape[1]
                center[2] = center[2]/raw.shape[2]
                
            
                class_id = classes.index(labels[idx])
                x_center = center[2] #center[1]
                y_center = center[1] #center[0]
                z_center = center[0] #center[2]
                
                #output = f'{class_id} {x_center} {y_center} {bb_size/raw.shape[0]} {bb_size/raw.shape[0]}\n'
                #output = f'{class_id} {z_center} {x_center} {y_center} {masks[zIndex].shape[0]/raw.shape[0]} {bb_size/raw.shape[1]} {bb_size/raw.shape[1]}\n'
                output = f'{class_id} {z_center} {x_center} {y_center} {cell_depth_z/raw.shape[0]} {cell_width_x/raw.shape[1]} {cell_height_y/raw.shape[2]}\n'
                f.write(output)
        #exit()
        count += 1
        
        if count%100 == 0:
            print(f'{count}\tof\t{len(files)}')
            
        if count == len(files):
            print(f'{count}\tof\t{len(files)}\nDONE')

print(time.time()-start)

#To construct mask file from labels .txt file to make sure they are correct in format and values
def multilabel_mask_maker(bbox_path: str, nifti_path: str, mask_path: str):
    """
    Makes nifti masks out of a YOLO label txt file.  Saves highest confidence mask for each class.
    Args:
        bbox_path: path to the YOLO label file.
        nifti_path: path to the corresponding nifti image file.
        mask_path: path to save the resultant mask as.
    """
    #print("in multilabel_mask_maker")
    f = open(bbox_path, 'r')
    label = list(filter(None, f.read().split('\n')))  # filtering out blank lines
    print("# of lines:",len(label))
    # load nifti
    nifti = nib.load(nifti_path)
    nifti_array = np.array(nifti.dataobj)
    # mask_array = np.zeros_like(nifti_array)
    print("shape of nifti:",nifti_array.shape)
    # might need to flip order of height and width...
    #height = nifti_array.shape[0]
    height = nifti_array.shape[1]
    #width = nifti_array.shape[1]
    width = nifti_array.shape[0]
    depth = nifti_array.shape[2]

    #box_dict = {}
    boxes = []
    count = 1
    for target in label:
        #cls, z, x, y, d, w, h, conf = target.split(' ')
        cls, z, x, y, d, w, h = target.split(' ')
        cls = int(cls)
        z = float(z)
        x = float(x)
        y = float(y)
        d = float(d)
        w = float(w)
        h = float(h)
        #conf = float(conf)

        #if cls not in box_dict.keys() or box_dict[cls][-1] < conf:
        #    box_dict[cls] = z, x, y, d, w, h, conf
        boxes.append((cls,x,y,z,w,d,h))
    
    mask_array = np.zeros_like(nifti_array)   # ----
    print("shape of masks array: ",mask_array.shape)
    print("# of boxes: ",len(boxes))
    #for cls in box_dict.keys():                    |
    for box in boxes:                            #  |
        # create empty mask                         |
        #mask_array = np.zeros_like(nifti_array)  --
        #z, x, y, d, w, h, conf = box_dict[cls]
        #cls,z, x, y, d, w, h = box
        cls,x, y,z, w, d, h = box
        z_center = z * depth
        x_center = x * width
        y_center = y * height
        z_length = d * depth
        x_length = w * width
        y_length = h * height

        min_z = int(math.floor(z_center - z_length / 2))
        max_z = int(math.ceil(z_center + z_length / 2))
        min_x = int(math.floor(x_center - x_length / 2))
        max_x = int(math.ceil(x_center + x_length / 2))
        min_y = int(math.floor(y_center - y_length / 2))
        max_y = int(math.ceil(y_center + y_length / 2))

        min_z = max(0, min_z)
        max_z = min(depth, max_z)
        min_y = max(0, min_y)
        max_y = min(height, max_y)
        min_x = max(0, min_x)
        max_x = min(width, max_x)

        mask_array[min_x:max_x+1, min_y:max_y+1, min_z:max_z+1] = count #1
        count += 1

        #cls_mask_path = mask_path.split('.')[0] + '_' + str(cls) + '.nii.gz'
        #mask_nifti = nib.Nifti1Image(mask_array, nifti.affine)
        #nib.save(mask_nifti, cls_mask_path)
    #cls_mask_path = mask_path.split('.')[0] + '_' + str(cls) + '.nii.gz'
    cls_mask_path = mask_path.split('.')[0] + '_' + 'pred' + '.nii.gz'
    print("unique values in mask_array : ", np.unique(mask_array))
    mask_nifti = nib.Nifti1Image(mask_array, nifti.affine)
    print("mask_nifti shape:",mask_nifti.shape)
    nib.save(mask_nifti, cls_mask_path)
    
#multilabel_mask_maker(r'S:\vivek\cellpose_3D\masked_images_with_img\annotation_results\cellpose_3D\labels\train\0.txt',r'S:\vivek\cellpose_3D\masked_images_with_img\annotation_results\cellpose_3D\images\train\0.nii',r'S:\vivek\cellpose_3D\masked_images_with_img\annotation_results\cellpose_3D\images\train')
multilabel_mask_maker(r'S:\vivek\cellpose_3D\masked_images_with_img\annotation_results\cellpose_3D\labels\train\0.txt',r'S:\vivek\cellpose_3D\masked_images_with_img\annotation_results\cellpose_3D\images\train\0.nii',r'S:\vivek\cellpose_3D\masked_images_with_img\annotation_results\cellpose_3D\images\train')

#%% save train and test files
#np.save(f'/home/nel/NEL-LAB Dropbox/NEL/Datasets/smart_micro/datasets/YOLO_train_files_{datetime.datetime.now().strftime("%m%d")}', train_files)
#np.save(f'/home/nel/NEL-LAB Dropbox/NEL/Datasets/smart_micro/datasets/YOLO_test_files_{datetime.datetime.now().strftime("%m%d")}', test_files)
