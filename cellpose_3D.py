import numpy as np
import matplotlib.pyplot as plt
from cellpose import models, io
from cellpose.io import imread
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.ndimage as ndimage
import cv2
from scipy.ndimage import find_objects
import os
import shutil
import tifffile

def crop_3d_tiff(image_path, crop_size, output_dir,index=1):
    """Crops a given 3D TIFF image into smaller images based on crop size. 
    If first value of crop_size tuple is '*', this skips cropping along the z-axis.

    Args:
        image_path (str): The path to the 3D TIFF image to be cropped.
        crop_size (tuple): A tuple of integers specifying the size of each crop along each dimension.   
        output_dir (str): The path to the directory where the cropped images will be saved.
        index (int): The index of the image i.e 1st image or 2nd image etc to difrentiate between crops from the name. Default is 1.
    """
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load the 3D TIFF image
    image = tifffile.imread(image_path)
    print(image.shape)
    for z in range(image.shape[0]):
        print(image[z,:,:].shape)
        break
    #exit()
    # Get the dimensions of the image
    depth, height, width = image.shape

    # Calculate the number of crops in each dimension
    if crop_size[0] != '*':
        num_crops_z = depth // crop_size[0]
        num_crops_y = height // crop_size[1]
        num_crops_x = width // crop_size[2]

        # Loop through each crop
        for z in range(num_crops_z):
            for y in range(num_crops_y):
                for x in range(num_crops_x):
                    # Calculate the crop boundaries
                    z_start = z * crop_size[0]
                    z_end = z_start + crop_size[0]
                    y_start = y * crop_size[1]
                    y_end = y_start + crop_size[1]
                    x_start = x * crop_size[2]
                    x_end = x_start + crop_size[2]

                    # Extract the crop
                    crop = image[z_start:z_end, y_start:y_end, x_start:x_end]

                    # Save the crop
                    output_path = f"{output_dir}/crop_{index}_{z}_{y}_{x}.tif"
                    tifffile.imwrite(output_path, crop)
    else:
        num_crops_y = height // crop_size[1]
        num_crops_x = width // crop_size[2]

        # Loop through each crop
        
        for y in range(num_crops_y):
            for x in range(num_crops_x):
                # Calculate the crop boundaries
                #z_start = z * crop_size[0]
                #z_end = z_start + crop_size[0]
                y_start = y * crop_size[1]
                y_end = y_start + crop_size[1]
                x_start = x * crop_size[2]
                x_end = x_start + crop_size[2]

                # Extract the crop
                crop = image[:, y_start:y_end, x_start:x_end]

                # Save the crop
                output_path = f"{output_dir}/crop_{index}_{depth}_{y}_{x}.tif"
                tifffile.imwrite(output_path, crop)

#Function copied from cellpose code. Needed to modify mask_flows_to_seg
def normalize99(Y, lower=1, upper=99, copy=True, downsample=False):
    """
    Normalize the image so that 0.0 corresponds to the 1st percentile and 1.0 corresponds to the 99th percentile.

    Args:
        Y (ndarray): The input image (for downsample, use [Ly x Lx] or [Lz x Ly x Lx]).
        lower (int, optional): The lower percentile. Defaults to 1.
        upper (int, optional): The upper percentile. Defaults to 99.
        copy (bool, optional): Whether to create a copy of the input image. Defaults to True.
        downsample (bool, optional): Whether to downsample image to compute percentiles. Defaults to False.

    Returns:
        ndarray: The normalized image.
    """
    X = Y.copy() if copy else Y
    X = X.astype("float32") if X.dtype!="float64" and X.dtype!="float32" else X
    if downsample and X.size > 224**3:
        nskip = [max(1, X.shape[i] // 224) for i in range(X.ndim)]
        nskip[0] = max(1, X.shape[0] // 50) if X.ndim == 3 else nskip[0]
        slc = tuple([slice(0, X.shape[i], nskip[i]) for i in range(X.ndim)])
        x01 = np.percentile(X[slc], lower)
        x99 = np.percentile(X[slc], upper)
    else:
        x01 = np.percentile(X, lower)
        x99 = np.percentile(X, upper)
    if x99 - x01 > 1e-3:
        X -= x01 
        X /= (x99 - x01)
    else:
        X[:] = 0
    return X

#Function copied from cellpose code. Needed to modify mask_flows_to_seg
def masks_to_outlines(masks):
    """Get outlines of masks as a 0-1 array.

    Args:
        masks (int, 2D or 3D array): Size [Ly x Lx] or [Lz x Ly x Lx], where 0=NO masks and 1,2,...=mask labels.

    Returns:
        outlines (2D or 3D array): Size [Ly x Lx] or [Lz x Ly x Lx], where True pixels are outlines.
    """
    if masks.ndim > 3 or masks.ndim < 2:
        raise ValueError("masks_to_outlines takes 2D or 3D array, not %dD array" %
                         masks.ndim)
    outlines = np.zeros(masks.shape, bool)

    if masks.ndim == 3:
        for i in range(masks.shape[0]):
            outlines[i] = masks_to_outlines(masks[i])
        return outlines
    else:
        slices = find_objects(masks.astype(int))
        for i, si in enumerate(slices):
            if si is not None:
                sr, sc = si
                mask = (masks[sr, sc] == (i + 1)).astype(np.uint8)
                contours = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_NONE)
                pvc, pvr = np.concatenate(contours[-2], axis=0).squeeze().T
                vr, vc = pvr + sr.start, pvc + sc.start
                outlines[vr, vc] = 1
        return outlines

#cellpose function modified to include target directory to save outputs and include raw image data in saved file,remove other fields
def masks_flows_to_seg(images, masks, flows, file_names, diams=30., channels=None,
                       imgs_restore=None, restore_type=None, ratio=1., target_dir=''):
    """Save output of model eval to be loaded in GUI.

    Can be list output (run on multiple images) or single output (run on single image).

    Saved to file_names[k]+"_seg.npy".
    
    Args:
        images (list): Images input into cellpose.
        masks (list): Masks output from Cellpose.eval, where 0=NO masks; 1,2,...=mask labels.
        flows (list): Flows output from Cellpose.eval.
        file_names (list, str): Names of files of images.
        diams (float array): Diameters used to run Cellpose. Defaults to 30.
        channels (list, int, optional): Channels used to run Cellpose. Defaults to None.

    Returns:
        None
    """

    if channels is None:
        channels = [0, 0]

    if isinstance(masks, list):
        if not isinstance(diams, (list, np.ndarray)):
            diams = diams * np.ones(len(masks), np.float32)
        if imgs_restore is None:
            imgs_restore = [None] * len(masks)
        if isinstance(file_names, str):
            file_names = [file_names] * len(masks)
        for k, [image, mask, flow, diam, file_name, img_restore
               ] in enumerate(zip(images, masks, flows, diams, file_names,
                                  imgs_restore)):
            channels_img = channels
            if channels_img is not None and len(channels) > 2:
                channels_img = channels[k]
            masks_flows_to_seg(image, mask, flow, file_name, diams=diam,
                               channels=channels_img, imgs_restore=img_restore,
                               restore_type=restore_type, ratio=ratio,target_dir=target_dir)
        return

    if len(channels) == 1:
        channels = channels[0]

    flowi = []
    if flows[0].ndim == 3:
        Ly, Lx = masks.shape[-2:]
        flowi.append(
            cv2.resize(flows[0], (Lx, Ly), interpolation=cv2.INTER_NEAREST)[np.newaxis,
                                                                            ...])
    else:
        flowi.append(flows[0])

    if flows[0].ndim == 3:
        #cellprob = (np.clip(transforms.normalize99(flows[2]), 0, 1) * 255).astype(
        #    np.uint8)
        cellprob = (np.clip(normalize99(flows[2]), 0, 1) * 255).astype(
            np.uint8)
        cellprob = cv2.resize(cellprob, (Lx, Ly), interpolation=cv2.INTER_NEAREST)
        flowi.append(cellprob[np.newaxis, ...])
        flowi.append(np.zeros(flows[0].shape, dtype=np.uint8))
        flowi[-1] = flowi[-1][np.newaxis, ...]
    else:
        flowi.append(
            #(np.clip(transforms.normalize99(flows[2]), 0, 1) * 255).astype(np.uint8))
            (np.clip(normalize99(flows[2]), 0, 1) * 255).astype(np.uint8))
        flowi.append((flows[1][0] / 10 * 127 + 127).astype(np.uint8))
    if len(flows) > 2:
        if len(flows) > 3:
            flowi.append(flows[3])
        else:
            flowi.append([])
        flowi.append(np.concatenate((flows[1], flows[2][np.newaxis, ...]), axis=0))
    outlines = masks * masks_to_outlines(masks) #utils.masks_to_outlines(masks)
    base = os.path.splitext(file_names)[0]

    dat = {
        "outlines":
            outlines.astype(np.uint16) if outlines.max() < 2**16 -
            1 else outlines.astype(np.uint32),
        "masks":
            masks.astype(np.uint16) if outlines.max() < 2**16 -
            1 else masks.astype(np.uint32),
        "chan_choose":
            channels,
        "ismanual":
            np.zeros(masks.max(), bool),
        "filename":
            file_names,
        "flows":
            flowi,
        "diameter":
            diams,
        "img":
            images.astype(np.uint16) if images.max() < 2**16 -
            1 else images.astype(np.uint32),
    }
    if restore_type is not None and imgs_restore is not None:
        dat["restore"] = restore_type
        dat["ratio"] = ratio
        dat["img_restore"] = imgs_restore
    target = os.path.join(target_dir, base + "_seg.npy")
    np.save(target,dat)
    #np.save(target_dir + base + "_seg.npy", dat)

image_path = r'S:\vivek\cellpose_3D\Scan_Iter_0000_CamA_ch0_CAM1_stack0000_488nm_0000000msec_0001003872msecAbs_resample_3_3_3.tif'
crop_size = ('*',352,352)
output_image_dir = r'S:\vivek\cellpose_3D\cropped_images'
output_masks_dir = r'S:\vivek\cellpose_3D\masked_images'
output_masks_img_dir_test = r'S:\vivek\cellpose_3D\masked_images_with_img_custom_func_test'

crop_3d_tiff(r'S:\vivek\cellpose_3D\Scan_Iter_0000_CamA_ch0_CAM1_stack0000_488nm_0000000msec_0001003872msecAbs_resample_3_3_3.tif',('*',352,352),r'S:\vivek\cellpose_3D\cropped_images',index=1)

print(output_masks_img_dir_test)

if not os.path.exists(output_masks_dir):
        os.makedirs(output_masks_dir)
#model = models.Cellpose(model_type='cyto2')
model = models.Cellpose(gpu=True, model_type='cyto')

#files = [r'S:\VGA\2024_03_08_LLSM_ExM\test_run_medium_512pxl_25um_2\matlab_stitch\matlab_decon_conventional\DSR\Scan_Iter_0000_CamA_ch0_CAM1_stack0000_488nm_0000000msec_0001003872msecAbs_resample_3_3_3_cellpose_sample.tif']
files = os.listdir(output_image_dir)
print("no. of files: ",len(files))

imgs = [imread(output_image_dir+'\\' +f) for f in files]
#nimg = len(imgs)
#print("no. of images: ",nimg)
print(imgs[0].shape)
print("model")
channels = [[0,0]]
i= 1

for file in files:
    #print("file: ",file)
    img = imread(output_image_dir+'\\' +file)
    #print(img.shape)
    masks, flows, styles, diams = model.eval(img, diameter=70, channels=channels,z_axis=0,do_3D=True,cellprob_threshold=2.0,min_size = 12000)
    #print("over")
    #io.masks_flows_to_seg(img, masks, flows, file, diams,channels)
    masks_flows_to_seg(img, masks, flows, file, diams,channels,target_dir=output_masks_img_dir_test)
    
    print("done: ",i)
    i += 1



