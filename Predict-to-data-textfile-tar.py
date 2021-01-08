#!/usr/bin/env python
# coding: utf-8

import os
import os.path
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import subprocess

from tqdm import tqdm
import skimage.io
import skimage.morphology

import tensorflow as tf
import keras
import utils.dirtools
import utils.metrics
import utils.model_builder
import sys
from skimage import img_as_bool, img_as_ubyte

import keras.layers
import keras.models
import tensorflow as tf

import tarfile
import utils.preprocess
import utils.model_builder


input_dir = '/'.join(os.path.abspath(sys.argv[1]).split("/")[0:-1]) + "/"

plate_name = sys.argv[1].split("/")[-1].split(".")[0]

input_dir = input_dir + plate_name + "/"

tarfile = tarfile.open(sys.argv[1])

nuclei_model_file = sys.argv[2]

cell_model_file = sys.argv[3]

lysosome_model_file = sys.argv[4]

savepath = input_dir + plate_name + '_predict_plots/'
os.makedirs(savepath, exist_ok = True)

png_dir =  input_dir + plate_name + '_png/'
os.makedirs(png_dir, exist_ok = True)

resultfile = open(input_dir + plate_name + '.txt', "w")

resultfile.write("Image\tAverage_nuclei_size\tNr_of_nuclei\tBright_parts\tAverage_nuclei_intensity\tAverage_cell_area\tLysosomes\n")
resultfile.close()
save_plots = True

# If nuclei model is a 3 channel model use 3 as input_channels and Merge=True.
#input_channels = 3
#Merge = True

input_channels = 1
Merge = False


# Create a list with all image names. 
ch_list = list(tarfile.getnames())
# filter list to only keep ch0 images in list.
ch0_list = list(filter(lambda x:"0.C01" in x, ch_list))

# Configuration to run on GPU
configuration = tf.compat.v1.ConfigProto()
configuration.gpu_options.allow_growth = True
#configuration.gpu_options.visible_device_list = "0"

session = tf.compat.v1.Session(config = configuration)

# apply session
tf.compat.v1.keras.backend.set_session(session)

# load reference image to know the image shapes

ref_C01 = input_dir + ch0_list[0]
tarfile.extract(ch0_list[0], input_dir) 
ref_png = png_dir + ch0_list[0].split("/")[-1][:-4] + ".png"

ref_im = utils.preprocess.bfconvert(ref_C01, ref_png)

dim1 = ref_im.shape[0]
dim2 = ref_im.shape[1]

nuclei_model = utils.model_builder.get_model_3_class(dim1, dim2, input_channels)
nuclei_model.load_weights(nuclei_model_file)

cell_model = utils.model_builder.get_model_3_class(dim1,dim2, 1)
cell_model.load_weights(cell_model_file)


lysosome_model = utils.model_builder.get_model_3_class(dim1,dim2,1)
lysosome_model.load_weights(lysosome_model_file)

for image in tqdm(ch0_list):
    # Opening and appending the resultfile so that data is stored during the process and will not be lost in case of crash.
    resultfile = open(input_dir + plate_name + '.txt', "a")
    
    ch_1_c01 = input_dir + image[:-5] + "1.C01"
    ch_1_png = png_dir + image.split("/")[-1][:-5] + "1.png"
    ch_0_c01 = input_dir + image
    ch_0_png = png_dir + image.split("/")[-1][:-4] + ".png"
    
    # Extract ch0 and ch1 image    
    tarfile.extract(image[:-5] + "1.C01", input_dir)
    tarfile.extract(image, input_dir)

    # Converting C01 image to png.
    ch_0_im = utils.preprocess.bfconvert(ch_0_c01, ch_0_png)
    ch_1_im = utils.preprocess.bfconvert(ch_1_c01, ch_1_png)

    ##### NORMALIZING IMAGES #####
    
    # Saving raw png 8 bit to use for bright
    ch_0_raw = ch_0_im.copy()
    ch_0_raw = img_as_ubyte(ch_0_raw)
    ch_1_raw = ch_1_im.copy()
    ch_1_raw = img_as_ubyte(ch_1_raw)

    ch_0_im = utils.preprocess.normalize(ch_0_im)
    ch_1_im = utils.preprocess.normalize(ch_1_im)
       
    ##### MERGING IMAGES #####
    # Merging images if merged model is used (e.g. 34 where ch0 and 2 are merged and 3d channel is black)
    if Merge == True:
        ch_2_im = np.zeros(ch_0_im.shape)        # Creating a zero image as 3d channel.
        ch_0_im = np.dstack((ch_0_im, ch_1_im, ch_2_im))
    
    ##### PREDICTION #####
    
    ch_0_im = ch_0_im.reshape((-1,dim1,dim2,input_channels))
    ch_1_im = ch_1_im.reshape((-1,dim1,dim2,1))
    ch_0_im = ch_0_im/255
    ch_1_im = ch_1_im/255
    ch0_prob = nuclei_model.predict(ch_0_im)[0]
    ch1_prob = cell_model.predict(ch_1_im)[0]
    lysosome_prob = lysosome_model.predict(ch_1_im)[0]
    
    ##### NUCLEI COUNT AND SIZE #####
    ch0_pred = utils.preprocess.probmap_to_pred(ch0_prob, 1)    
    ch0_label = utils.preprocess.pred_to_label(ch0_pred, 100)  
    ch0_label = skimage.morphology.label(ch0_label)
    
    #Increase size of objects. This is done because the border of the predicted objects are covering part of interior.
    struct = skimage.morphology.square(3)
    ch0_prediction = skimage.morphology.dilation(ch0_label, struct)

    #Retrieve object pixel values, and corresponding count (sizes of objects.)
    nuclei_list, sizes= np.unique(ch0_prediction, return_counts = True)
    nuclei = len(nuclei_list)-1
    
    #Get average nuclei size
    avg_nuc_size = sum(sizes[1:])/nuclei
    
    # Get avg nuclei intensity
    avg_intensity = utils.preprocess.get_nuclei_intensity(ch_0_raw,ch0_prediction)

    ##### GET LYSOSOME COUNT
    lysosome_pred = utils.preprocess.probmap_to_pred(lysosome_prob,1)
    
    lysosome_label = utils.preprocess.pred_to_label(lysosome_pred,2, cell_label = 2)

    lysosomes = len(np.unique(lysosome_label)) - 1
    
    #Making lysosomes larger to show on plot
    struct = skimage.morphology.square(10)
    lysosome_label = skimage.morphology.dilation(lysosome_label, struct)
    
    ##### FIND BRIGHT PARTS #####
    
    thresh = min((avg_intensity * 2.4), 240)
    brightparts = np.zeros(ch_0_raw.shape)
    brightparts[ch_0_raw > thresh] = 255
    brightparts = img_as_bool(brightparts)
    
    # Removing small objects, to avoid getting false positives in cells with a lot of "small bright spots"
    brightparts = skimage.morphology.remove_small_objects(brightparts, min_size=10)
    
    # Increase size of objects so that bright parts really close to eachother (most likely belonging to the same cell) are merged.
    struct = skimage.morphology.square(10)
    brightparts = skimage.morphology.dilation(brightparts, struct)
    
    # retrieving the separate objects (surrounded with background counts as one unique object)
    brightlabels = skimage.morphology.label(brightparts)
    
    # Counts the unique pixel values. -1 because of the unique background color that is not an object.
    bright_parts = len(np.unique(brightlabels))-1

    
    #### GET CELL AREA ####
    
    ch1_pred = utils.preprocess.probmap_to_pred(ch1_prob, 1)
    area_im = utils.preprocess.get_cell_area(ch1_pred)
    cell_area = np.unique(area_im, return_counts = True)[1][1]
    avg_cell_area = cell_area/nuclei
    
    
    
    resultfile.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(image,avg_nuc_size,nuclei,bright_parts, avg_intensity, avg_cell_area, lysosomes))
    resultfile.close()

    #### PLOT RESULTS ####
    if save_plots:    
        fig, ax = plt.subplots(2, 4, figsize=(30,20))
        ax[0][0].imshow(ch_0_raw)
        ax[0][0].set_title("Channel 0 image")
        ax[0][1].imshow(ch_1_raw)
        ax[0][1].set_title("Channel 1 image")
        ax[0][2].imshow(ch0_prediction)
        ax[0][2].set_title("Predicted objects: "+str(nuclei))
        ax[0][3].imshow(brightlabels)
        ax[0][3].set_title("Predicted bright parts: " + str(bright_parts))
        ax[1][0].imshow(area_im)
        ax[1][0].set_title("Predicted cell Area")
        ax[1][1].imshow(ch1_prob)
        ax[1][1].set_title("Probabilitymap Cell")
        ax[1][2].imshow(ch0_prob)
        ax[1][2].set_title("Probabilitymap Nuclei")    
        ax[1][3].imshow(lysosome_label)
        ax[1][3].set_title("Lysosomes" + str(lysosomes))
        plt.savefig(savepath + image.split("/")[-1][:-4] + ".png")
        plt.close(fig)
