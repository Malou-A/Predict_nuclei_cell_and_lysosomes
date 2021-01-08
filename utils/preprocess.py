import numpy as np
import skimage
import skimage.io
import skimage.morphology
from skimage import img_as_ubyte, img_as_bool
import subprocess

def normalize(orig_img):

    percentile = 99.9
    high = np.percentile(orig_img, percentile)
    low = np.percentile(orig_img, 100-percentile)

    img = np.minimum(high, orig_img)
    img = np.maximum(low, img)

    img = (img - low) / (high - low) # gives float64, thus cast to 8 bit later
    
    img = skimage.img_as_ubyte(img)
    return(img)

def pred_to_label(pred, cell_min_size, cell_label=1):
    # Only marks interior of cells (cell_label = 1 is interior, cell_label = 2 is boundary)
    cell=(pred == cell_label)
    # fix cells
    cell = skimage.morphology.remove_small_holes(cell, area_threshold=500)
    cell = skimage.morphology.remove_small_objects(cell, min_size=cell_min_size)
    
    # label cells only
    [label, num] = skimage.morphology.label(cell, return_num=True)
    return label

def probmap_to_pred(probmap, boundary_boost_factor):
    # we need to boost the boundary class to make it more visible
    # this shrinks the cells a little bit but avoids undersegmentation
    pred = np.argmax(probmap * [1, 1, boundary_boost_factor], -1)
    
    return pred

def get_nuclei_intensity(img, annot):
    annot_bool = np.zeros(annot.shape)
    annot_bool[annot>0] = True
    annot_bool = img_as_bool(annot_bool)
    newim = img.copy()
    newim[annot_bool==False] = 0
    pixels, value = np.unique(newim, return_counts = True)
    val = 0
    for i,pixel in enumerate(pixels):
        val += pixel*value[i]
    avg_intensity = val/sum(value[1:])
    return(avg_intensity)

def get_cell_area(img):
    
    cell = (img == 1) + (img == 2)
    cell = skimage.morphology.remove_small_holes(cell, area_threshold=500)
    cell = skimage.morphology.remove_small_objects(cell, min_size=200)
    return(cell)


def bfconvert(orig_im, converted):
    subprocess.run(['bfconvert', '-overwrite', '-nogroup',orig_im,converted],stdout = subprocess.PIPE, stderr = subprocess.DEVNULL) #Runs bftools which needs to be preinstalled, output to DEVNULL.
    subprocess.run(['convert', orig_im, '-auto-level', '-depth', '8', '-define', 'quantum:format=unsigned', '-type', 'grayscale', converted],stdout = subprocess.PIPE, stderr = subprocess.DEVNULL) #Convert images to 16-bits tiff images.
    im = skimage.io.imread(converted)
    return(im)

