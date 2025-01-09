#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 18:56:25 2021

@author: jimmytabet
"""

############################## SET UP ANNOTATOR ###############################

# imports
import sys, os, glob, cv2, re
import numpy as np
from scipy import ndimage
import logging
#set to debug to get debugging print statements
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # silence TensorFlow error message about not being optimized...

#----------------------------------USER INPUT---------------------------------#

# if working in terminal, run:    'python /path/to/annotation_gui.py /path/to/data'
#   or cd to data folder and run: 'python /path/to/annotation_gui.py $(pwd)'
#   or edit path_to_data and run: 'python /path/to/annotation_gui.py'

# boolean to use Jupyter Notebook
# if working in JN, run '%load annotation_gui.py' in new .ipynb file and change JN = True
JN = False

# initials of annotator
ANNOTATOR_INITIALS = 'JT'

# boolean to use automatic neural network filter
nn_filter = True
# path to neural network filter (should be *.hdf5)
path_to_nn_filter = r"S:\vivek\\smartLLSM Demo Data\smartLLSM Demo Data\annotator_filter.hdf5"
# threshold confidence to classify "unique" cells/automatically filter tiles
filter_thresh = 0.6 #0.7

# boolean to use pre-filtered file list
prefiltered = False
# path to pre-filtered file list
path_to_prefiltered_files = ''

# path to data (optionally passed in terminal - use '$(pwd)' to pass pwd)
path_to_data = r"S:\vivek\cellpose_3D\masked_images_with_img_custom_func_test"  #"S:\vivek\\smartLLSM Demo Data\smartLLSM Demo Data\annotation_demo\annotation_demo"

# labels dictionary
#!!!!!!!!!!!! WARNING, KEYS MUST BE UNIQUE AND NOT CONTAIN 'temp' !!!!!!!!!!!!#
labels_dict = {-1: 'edge',     # automatically detected
                0: 'blurry',
                1: 'interphase',
                2: 'prophase',
                3: 'prometaphase',
                4: 'metaphase',
                5: 'anaphase',
                6: 'telophase',
                7: 'junk',
                8: 'TBD',
                9: 'other'}

# show label dictionary key(s)
show_label = ['l','d']

# manually assign edge
man_edge = ['-']

# back key(s)
back_key = ['b']

# exit key(s)
exit_key = ['q']

# threshold percentage of area needed to see edge cell/automatically assign edge cell
edge_area_thresh = 0.3 #0.7

# half of size to show cell
cell_half_size = 50 #100

# half of size to show tile (must be < 800/2=400)
# set <= 0 to display whole tile
show_half_size = 0 # < 400

#-----------------------------------------------------------------------------#

# get edge key for automatic edge assignment
edge_key = [k for k,v in labels_dict.items() if v == 'edge'][0]

# set path_to_data if run in terminal
if len(sys.argv) > 1 and not JN:
    path_to_data = sys.argv[1]
else:
    path_to_data = path_to_data

# check that given path is actually a folder
while not os.path.isdir(path_to_data):
    path_to_data = input('ERROR: '+path_to_data+' not found, try again\npath to data: ')
    path_to_data = os.path.abspath(path_to_data)

# change directory to data
os.chdir(path_to_data)

# create list of tiles to annotate (exclude 'finished' tiles)
if prefiltered:
    # check for matching pre-filtered list and data set
    files_num = re.search(r'data_[0-9]_', path_to_data).group()
    filtered_num = re.search(r'data_[0-9]_', path_to_prefiltered_files).group()
    if not files_num == filtered_num:
        ans = input(f'Potential mismatch between pre-filtered file list (\'{filtered_num}\') and data set (\'{files_num}\').\nEnter \'q\' to exit, or any other key to continue: ')
        if ans == 'q':
            raise ValueError('Potential mismatch between pre-filtered file list and data set')
    
    # load pre-filtered list
    prefiltered_files = np.load(path_to_prefiltered_files)
    tiles = [os.path.join(path_to_data, tile) for tile in prefiltered_files]
else:
    print('CREATING LIST OF FILES FOR ANNOTATION...')
    tiles = sorted(glob.glob(os.path.join(path_to_data,'**','*.npy'), recursive=True))
    # remove finished tiles
    tiles = [file for file in tiles if not '_finished' in file]
    # remove preprocessed file list
    tiles = [file for file in tiles if not '_preprocessed' in file]
    
    print("len tiles: ",len(tiles))
    print("sample tile  ",tiles[0])

# folder for annotation results
if 'data_' in path_to_data:
    results_folder = os.path.join(os.path.dirname(path_to_data), 'annotation_results')
else:
    results_folder = os.path.join(path_to_data, 'annotation_results')
print("results folder: ",results_folder)

# set up annotator if there are files to annotate
if tiles:
    # disable nn_filter if using prefiltered list
    if prefiltered:
        nn_filter = False
    
    # set up neural network and filtering function if used
    if nn_filter:
        print('LOADING NEURAL NETWORK FILTER...')
        import tensorflow as tf
        def nn_temp(a,b):return True
        model = tf.keras.models.load_model(path_to_nn_filter, custom_objects={'f1_metric':nn_temp})
        # set filter class/column
        output_classes = np.array(model.name.split('__temp__')) #
        #logging.debug("output classes: ",output_classes)
        filter_class = 'unique'
        while not filter_class in output_classes:
            filter_class = input('Invalid filter class, pick from the following: '\
                                 +str(output_classes)+'\n\tfilter class: ')
        
        filter_col = int(np.where(filter_class == output_classes)[0])
        #logging.debug(filter_col)
        # filtering function to determine if tile contains interesting cell
        def interesting(nn_model, raw_tile, masks_tile, thresh, thresh_col):
        
            # find number of identified cells
            #num_masks = np.max(masks_tile)
            num_masks = len(np.unique(masks))
            mask_id_values = np.unique(masks)
            
            # loop through each cell and check if interesting
            #for mask_id in range(1,num_masks+1):
            for mask_id in mask_id_values:
                if mask_id == 0:
                    continue
                temp =  masks_tile==mask_id
                #print("temp shape in interesting: ",temp.shape)
                #print("mask_id: ",mask_id)
                if temp.shape[0] == 0:
                    #return False
                    continue
                # find center of mass (as integer for indexing)
                center = ndimage.center_of_mass(masks_tile==mask_id)
                center = np.array(center).astype(int)
                logging.debug("center: %s",center)
                
                # create image to test for filtering with nn
                nn_half_size = nn_model.input_shape[1]//2
                logging.debug("nn_half_size: %s",nn_half_size)
                r1_o = center[0]-nn_half_size
                c1_o = center[1]-nn_half_size
                logging.debug("r1_o %s",r1_o)
                logging.debug("c1_o %s",c1_o)
                # find bounding box indices to fit in tile
                r1_nn = max(0, center[0]-nn_half_size)
                r2_nn = min(raw_tile.shape[0], center[0]+nn_half_size)
                c1_nn = max(0, center[1]-nn_half_size)
                c2_nn = min(raw_tile.shape[1], center[1]+nn_half_size)
                # pad new bounding box with constant value (mean, 0, etc.)
                nn_test = np.zeros([nn_half_size*2, nn_half_size*2])
                nn_test += raw_tile[masks_tile==0].mean().astype('int')
                logging.debug("r1_nn: %s",r1_nn)
                logging.debug("r2_nn: %s",r2_nn)
                logging.debug("c1_nn: %s",c1_nn)
                logging.debug("c2_nn: %s",c2_nn)
                
                # store original bb in new bb
                nn_test[r1_nn-r1_o:r2_nn-r1_o,c1_nn-c1_o:c2_nn-c1_o] = raw_tile[r1_nn:r2_nn,c1_nn:c2_nn]
        
                # normalize image
                mean, std = np.mean(nn_test), np.std(nn_test)
                nn_test = (nn_test-mean)/std
                # add dimension to input to model
                nn_test = nn_test.reshape(1,*nn_model.input_shape[1:])
                preds = nn_model.predict(nn_test).squeeze()
                
                # return True if interesting
                if preds[thresh_col] > thresh:
                    return True
                else:
                    
                    #print("continuing in interesting func")
                    continue
    
    # raise error if key conflict/duplicate keys found
    all_keys = [str(i) for i in [labels_dict.keys(), show_label, man_edge, back_key, exit_key] for i in i]
    key_names = ['labels_dict', 'show_label', 'man_edge', 'back_key', 'exit_key']
    if len(all_keys) != len(set(all_keys)):
        raise ValueError('Key conflict/duplicate keys found, check '+'/'.join(key_names)+' variables!')
    
    # make sure show_half_size behaves
    while show_half_size >= 400:
        print('WARNING: show_half_size (half of tile to show) must be less than 800/2 = 400', end='')
        show_half_size = input('enter new value for show_half_size (or 0 to show entire tile): ')
        isint = False
        while not isint:
            try:
                show_half_size = int(show_half_size)
                isint = True
            except ValueError:
                show_half_size = input('please enter integer: ')
    
    # if show_half_size is smaller than cell_half_size, revert to showing entire tile
    if 0 < show_half_size < cell_half_size:
        show_half_size = 0
        print('show_half_size too small, reverting to showing entire tile')
    
    # print annotator set up
    print()
    print('ANNOTATOR SET UP:')
    print()
    
    # data folder
    print('data folder (pwd):\n\t', os.getcwd())
    print()
    
    # print/create results folder (if does not exist)
    if not os.path.isdir(results_folder):
        os.makedirs(results_folder)
        print('created annotation results folder:\n\t', results_folder)
    else:
        print('annotation results folder:\n\t', results_folder)
    print()
    
    # files to annotate
    print('files ('+str(len(tiles))+' total):')
    short_name = [os.path.relpath(tile) for tile in tiles]
    for i in short_name[:5]:
        print('\t',i)
    if len(tiles) > 5:
        print('\t ...')
    print()

    # exit key(s)
    print('exit key(s):\n\t', exit_key)
    print()
    
    # back key(s)
    print('back key(s):\n\t', back_key)
    print()

    # manually assign edge
    print('manually assign edge key(s):\n\t', man_edge)
    print()

    # show label key(s)
    print('show label dictionary key(s):\n\t', show_label)
    print()

    # labels
    print('labels:')
    for k,v in labels_dict.items():
        if v=='edge':
            print('\t%3s: %-8s\t(automatically assigned)' %(k,v))
        else:
            print('\t%3s: %s' %(k,v))
    print()
    
############################### RUN ANNOTATOR #################################

# close all windows
cv2.destroyAllWindows()
cv2.waitKey(1)

################## LOOP THROUGH EVERY 800X800 TILE IN FOLDER ##################

# init exited to see if user exited annotator
exited = False
centers_file = r'S:\vivek\cellpose_3D\masked_images_with_img\centers'
for tile in [r"S:\vivek\cellpose_3D\masked_images_with_img_custom_func_test\crop_232_1_1_seg.npy"]: #tiles: #Each tile is 3D .npy file 5_7
    
    # load Cellpose data (raw image, masks, and outlines)
    if prefiltered:
        try:
            data = np.load(tile, allow_pickle=True).item()
        except FileNotFoundError:
            continue
    
    else:
        data = np.load(tile, allow_pickle=True).item()
    
    #Now loop for eaach 2-D XY slice across Z
    #Still z,y,x dim
    ''''
    raw_full = data['img']
    masks_full = data['masks']
    outlines_full = data['outlines']
    
    #For each mask_id, get the z-location of the middle slice of that mask_id volume
    labels_full = []
    print(np.unique(masks_full))
    visited = [0 for _ in range(masks_full.max()+1)]
    mid_indices = []
    num_masks = masks_full.max()
    print("number of masks: ",num_masks)
    '''
    #To test if number of labels doe not equal number of masks
    repeat = True
    while repeat:
        raw_full = data['img']
        masks_full = data['masks']
        outlines_full = data['outlines']
        
        #For each mask_id, get the z-location of the middle slice of that mask_id volume
        labels_full = []
        print(np.unique(masks_full))
        visited = [0 for _ in range(masks_full.max()+1)]
        mid_indices = []
        num_masks = masks_full.max()
        print("number of masks: ",num_masks)
        if num_masks < 2:
            print("Not enough cells present in this tile")
            file_name = os.path.basename(tile)
            position_path = os.path.dirname(tile)
            position_folder = os.path.basename(position_path)            
            data_folder = os.path.basename(os.path.dirname(position_path))
            annotated_data_path = os.path.join(results_folder,data_folder)
            results_folder = os.path.join(os.path.dirname(path_to_data), 'annotation_results')
            print("file_name: ",file_name)
            print("position_path: ",position_path)
            print("position_folder: ",position_folder)
            print("data_folder: ",data_folder)
            empty_tiles_path = os.path.join(results_folder, 'empty_tiles')
            #if not os.path.isdir(annotated_data_path):
            #    os.makedirs(annotated_data_path)
            #exit()
            break
        
        for idx, mask_id in enumerate(range(1,num_masks+1)):
            correct = False
            while not correct:
                conditionMet = (masks_full==mask_id)
                mask_indices  = np.nonzero(conditionMet)
                mid_index = mask_indices[0].min() + (mask_indices[0].max() - mask_indices[0].min())//2 #in Z
                #mid_indices.append(mid_index)
                z = mid_index
                raw = (raw_full*conditionMet)[mid_index]
                masks = (masks_full*conditionMet)[mid_index]
                outlines = (outlines_full*conditionMet)[mid_index]
                print("z slice: ",z)
                print("mask_id: ",mask_id)
                # if using filter, check for interesting cells in tile and skip if none found 
                #Do it for each slice now
                
                #The 2D slice is Y,X. we change this to X,Y
                #raw = raw.T
                #masks = masks.T 
                #2outline = outlines.T
                #exit()
                #if nn_filter and not interesting(model, raw, masks, filter_thresh, filter_col):
                #    print(os.path.relpath(tile)+f'{z} slice - FILTERED' + f"nn_filter - {nn_filter}")
                #    continue

                # print current tile
                print('-'*53)
                print('\''+os.path.relpath(tile).upper()+'\''+f'{z} slice')

                # init repeat and correct (to catch if number of labels does not equal number of cells)
                
                #correct = False
                
            ############# LOOP UNTIL NUMBER OF LABELS MATCHES NUMBER OF CELLS #############
                #while not correct:               
                # init cell label and list for cell labels in tile
                label = None
                #labels=[]
                #Since now the label index no longer matches with mask_id, use a dict instead to index on mask_id
            #labels = {}
                # init mask_id/cell number
                #mask_id = 1
            #    mask_id_index = 0
            #    mask_id = mask_id_values[mask_id_index]
        ####################### LOOP THROUGH EVERY CELL IN TILE #######################
                
                #while mask_id <=num_masks:
            #    while mask_id_index < num_masks:    
            #        mask_id = mask_id_values[mask_id_index]
            #        if mask_id == 0:
            #            mask_id_index += 1
            #            continue
                    #optimization : if the mask is aldready seen in previous slice since that label is present this can be skipped
            #        if visited[mask_id] != 0:
            #            labels[mask_id] = 'seen before'
            #            visited[mask_id] += 1
            #            mask_id_index += 1
            #            continue
                    
                    # print cell ID
                    #print('cell: %2d of %2d --> class: ' % (mask_id, num_masks), end='')
                    #print('cell mask ID: %2d. Cell %2d of %2d in this slice --> class: ' % (mask_id,mask_id_index+1, num_masks), end='')
                print('cell mask ID: %2d. Cell %2d of %2d in this 3D tile --> class: ' % (mask_id,mask_id, num_masks), end='')
                    # find center of mass (as integer for indexing)
                    #center = ndimage.center_of_mass(masks==mask_id)
                #center = ndimage.center_of_mass(masks==mask_id)
                center = ndimage.center_of_mass(masks)
                center = np.array(center).astype(int)
                    #print("mask_id: ",mask_id)
                    #print("center: ",center)
                    # find bounding box indices for showing isolated cell
                r1 = max(0, center[0]-cell_half_size)
                r2 = min(raw.shape[0], center[0]+cell_half_size)
                c1 = max(0, center[1]-cell_half_size)
                c2 = min(raw.shape[1], center[1]+cell_half_size)
                
                    # check bounding box to see if cell on edge            
                area_ratio = (r2-r1)*(c2-c1)/(cell_half_size*2)**2
            
        #-------------automatically classify as edge if too close to edge-------------#

                if area_ratio < edge_area_thresh:
                    label = edge_key
                # add 'auto' designation for when later trying to go back
                #labels.append(str(label) + 'auto')
                #labels[mask_id] = str(label) + 'auto'
                    labels_full.append(str(label) + 'auto')
                    visited[mask_id] += 1
                    print(labels_dict[label]+' (automatically assigned)')
                #mask_id += 1
                #mask_id_index += 1
                    #continue
                    correct = True
            #        break
                    # fix edge case if most of area is in frame
                else:
                    rfix = cell_half_size*2 - (r2-r1)
                    cfix = cell_half_size*2 - (c2-c1)
                    if r1 == 0: r2 += rfix
                    if r2 == raw.shape[0]: r1 -= rfix
                    if c1 == 0: c2 += cfix
                    if c2 == raw.shape[1]: c1 -= cfix

            # find bounding box indices for showing tile if given size
                if show_half_size:                
                    show_r1 = max(0, center[0]-show_half_size)
                    show_r2 = min(raw.shape[0], center[0]+show_half_size)
                    show_c1 = max(0, center[1]-show_half_size)
                    show_c2 = min(raw.shape[1], center[1]+show_half_size)
                    show_rfix = show_half_size*2 - (show_r2-show_r1)
                    show_cfix = show_half_size*2 - (show_c2-show_c1)
                    if show_r1 == 0: show_r2 += show_rfix
                    if show_r2 == raw.shape[0]: show_r1 -= show_rfix
                    if show_c1 == 0: show_c2 += show_cfix
                    if show_c2 == raw.shape[1]: show_c1 -= show_cfix
                else:
                    show_r1 = 0
                    show_r2 = raw.shape[0]
                    show_c1 = 0
                    show_c2 = raw.shape[1]
                    
        #--------------------------construct images to show---------------------------#

                    # copy raw to show outline
                raw_outline = raw.copy()
                raw_outline[outlines==mask_id] = raw.max()
                raw_outline = raw_outline[show_r1:show_r2,show_c1:show_c2]
            # copy raw to show isolated cell
                raw_isolate = raw.copy()
                #raw_isolate[masks!=mask_id] = 0
            # resize isolated cell to show with tile
                raw_isolate = raw_isolate[r1:r2,c1:c2]
                raw_isolate = cv2.resize(raw_isolate, raw_outline.shape)
            
            # shorthand for arranging images
                left_img = raw_outline
                right_img = raw_isolate
            
            # add white border between panels
                col = 5
                border = raw.max()*np.ones([left_img.shape[0], col])
                left_img = np.concatenate([left_img, border], axis=1)
            
            # stitch left and right windows together
                together = np.concatenate([left_img/left_img.max(), right_img/right_img.max()], axis=1)

        #---------------------------set up annotator window---------------------------#

                    # show cell number in window with extra info as necessary
                    #window = os.path.relpath(tile).upper()+': CELL '+str(mask_id)+' OF '+str(num_masks)
                print("here")
                print("repeat: ",repeat)
                window = os.path.relpath(tile).upper()+': CELL '+str(mask_id)+' OF '+str(num_masks)
                #if repeat:
                #    window = 'ERROR: NUMBER OF LABELS DID NOT MATCH NUMBER OF CELLS, REPEATING '+os.path.relpath(tile).upper()
                #    repeat = False
                #elif label == edge_key:
                if label == edge_key:
                    window += ' (previous cell(s) on edge)'
                    break    
                    # show annotator window and image
                cv2.namedWindow('Cell Annotator', cv2.WINDOW_AUTOSIZE)
                cv2.setWindowTitle('Cell Annotator', window)
                cv2.imshow('Cell Annotator', together)
                    
                    # cv2.resizeWindow('Cell Annotator', together.shape[1], together.shape[0]) 
            
        #----------------------------------annotate-----------------------------------#
                    
                    # init back (to allow for returning to previous cell)
                back = False
                    
                    # init valid to check if response is valid
                valid = False
                    
        ################# LOOP UNTIL VALID LABEL RESPONSE IS DETECTED #################

            #while not valid:
        
                # wait for user entry and store
                res = cv2.waitKey(0)
                label = chr(res)
                # convert label to integer if number
                try:
                    label = int(label)
                except ValueError:
                    label = label

        #----------------------------------exit key-----------------------------------#

                        # if exit key is pressed, activate exited condition and break out of valid loop
                if label in exit_key:
                    exited = True
                    print('\n')
                    print('-'*53)
                    print('EXITING ANNOTATOR...')
                    break
                        
        #---------------------------show labels dictionary----------------------------#

                elif label in show_label:
                    print('\n')
                    print('labels:')
                    for k,v in labels_dict.items():
                        if v=='edge':
                            print('\t%3s: %-8s\t(automatically assigned)' %(k,v))
                        else:
                            print('\t%3s: %s' %(k,v))
                    print()
                    print('manually assign edge key(s): ', man_edge)
                    print('back key(s): ', back_key)
                    print('exit key(s): ', exit_key)
                    print()
                    #print('cell: %2d of %2d --> class: ' % (mask_id, num_masks), end='')
                    print('cell: %2d of %2d --> class: ' % (mask_id, num_masks), end='')
                    '''
                            #----------------------------------back key-----------------------------------#                
                            
                            # if back key is pressed, attempt to go back one cell
                            elif label in back_key:
                                # reverse through labels to check if possible
                                for index, lab in enumerate(labels[::-1]):    
                                    # if possible, reset to that cell
                                    if lab != str(edge_key) + 'auto':
                                        # activate back condition and valid input
                                        back = True
                                        valid = True
                                        # reset cell number
                                        mask_id = len(labels) - index
                                        # reset labels
                                        labels = labels[:mask_id-1]
                                        print('BACK TO CELL', mask_id)
                                        break

                                # if not possible, raise error
                                if not back:
                                    print('UNABLE TO RETURN')
                                    print('cell: %2d of %2d --> class: ' % (mask_id, num_masks), end='')
                                    cv2.setWindowTitle('Cell Annotator', 'UNABLE TO RETURN') 
                            
                            #------------------------key not in labels dictionary-------------------------#
                            '''
                        # if label is not in labels dictionary, raise error
                elif label not in list(labels_dict.keys()) + man_edge:
                    print('unrecognized label')
                    #print('cell: %2d of %2d --> class: ' % (mask_id, num_masks), end='')
                    print('cell: %2d of %2d --> class: ' % (mask_id, num_masks), end='')
                    cv2.setWindowTitle('Cell Annotator', 'ERROR: UNRECOGNIZED LABEL')    
                        
        #--------------------------------valid label----------------------------------#

                        # add label if valid
                else:
                    # activate valid input
                    #valid = True
                    correct = True
                    if label in man_edge:
                        label = edge_key
                        print(labels_dict[label]+' (manually assigned)')
                    else:
                        print(labels_dict[label])
                    #labels.append(label)
                    #labels[mask_id] = label
                    labels_full.append(label)
                    visited[mask_id] += 1
            #--------------------------response to valid label----------------------------#

                        # if exit key is pressed in valid loop, break out of cell loop
            if label in exit_key:
                break
            # if back is triggered, reset to false and revert to previous cell
            elif back:
                back = False
            # continue to next cell if everything is fine
            #else:
                #mask_id += 1
            #    mask_id_index += 1
                
        #--------------check if number of labels match number of cells----------------#
                
                # if exit key is pressed in cell loop, break out of correct loop   
        if label in exit_key:
            break
            
                # raise error if there is not a label for every cell in tile and repeat tile annotation
        if len(labels_full) != num_masks:  #add -1 because we always skip mask_id 0
            # activate repeat condition
            print("len(labels_full) ",len(labels_full))
            repeat = True
            print()
            print('ERROR: number of labels does not match number of cells, repeating \''+os.path.relpath(tile)+'\' ...')
            print()
        
        # fix and save labels and move files if all is good
        else:            
            # activate correct condition
        #    correct = True
                    
                    # fix automatic edge label
                    #for i,lab in enumerate(labels):
            #for i,lab in labels.items():
            for i,lab in enumerate(labels_full):
                if lab == str(edge_key) + 'auto':
                    #labels[i] = edge_key
                    labels_full[i] = edge_key
                else:
                    pass
                
        #for i in mask_id_values:
        #    if visited[i] == 1:
        #        labels_full.append(labels[i])
                    
            
            #print(f"Slize {z}")
            #print("labels for this slice: ",labels)
            #print("labels full: ",labels_full)
            #print([(i, visited[i]) for i in range(len(visited))])

        
            # convert labels to object np.array (for proper npz saving) if any labels are strings
            #if any([isinstance(i,str) for i in labels]):
                #labels = np.array(labels, dtype='object')
        if any([isinstance(i,str) for i in labels_full]):
            labels_full = np.array(labels_full, dtype='object')
        
        repeat  = False
        # save and move files
        # get file name, position/data paths, and annotated data path
        file_name = os.path.basename(tile)
        position_path = os.path.dirname(tile)
        position_folder = os.path.basename(position_path)            
        data_folder = os.path.basename(os.path.dirname(position_path))
        annotated_data_path = os.path.join(results_folder,data_folder)
        
        # get finished and results paths
        results_path = os.path.join(annotated_data_path, position_folder+'_results', file_name[:-7]+'annotated.npz')
        finished_path = os.path.join(position_path+'_finished', file_name)
        
        # create results and finished folders if not already created
        # new annotated data folder
        if not os.path.isdir(annotated_data_path):
            os.makedirs(annotated_data_path)
        # new annotated position folder
        if not os.path.isdir(os.path.dirname(results_path)):
            os.makedirs(os.path.dirname(results_path))
        # finished position folder
        if not os.path.isdir(os.path.dirname(finished_path)):
            os.makedirs(os.path.dirname(finished_path))

        # save annotated info for tile once all cells have been labeled
        '''save_dict = {'raw': raw,
                        'masks': masks,
                        #'labels': [labels_dict[j] for j in labels],
                        'labels': [labels_dict[j] for j in labels_full],
                        #f'labels_{ANNOTATOR_INITIALS}': [labels_dict[j] for j in labels],
                        f'labels_{ANNOTATOR_INITIALS}': [labels_dict[j] for j in labels_full],
                        'labels_dict': labels_dict,
                        'confirmed': []
                        }'''
        save_dict = {'raw': raw_full,
                    'masks': masks_full,
                        #'labels': [labels_dict[j] for j in labels],
                    'labels': [labels_dict[j] for j in labels_full],
                        #f'labels_{ANNOTATOR_INITIALS}': [labels_dict[j] for j in labels],
                    f'labels_{ANNOTATOR_INITIALS}': [labels_dict[j] for j in labels_full],
                    'labels_dict': labels_dict,
                    'confirmed': []
                        }
        
        np.savez(results_path, **save_dict)
        # move file to completed folder
        os.replace(tile, finished_path)

        # print paths
        short_path = os.path.relpath(tile)
        short_results = os.path.relpath(results_path)     
        short_finished = os.path.relpath(finished_path)            
        print()
        print('FINISHED --> \''+short_path+'\'')
        print(' RESULTS --> \''+short_results+'\'')
        print('    TILE --> \''+short_finished+'\'')
        print()
        print("Onto next tile")
    #---------------------------------(exit key)----------------------------------#

        # if exit key is pressed in correct loop, break out of tile loop   
        if label in exit_key:
            break
    
############################### CLOSE ANNOTATOR ###############################
    
# close all windows
cv2.destroyAllWindows()
cv2.waitKey(1)

# print exit message
if not exited:
    print('-'*53)
    print('NO FILES FOUND/ALL FILES FINISHED OR FILTERED IN:\n'+path_to_data)
else:
    print('ANNOTATOR EXITED')
