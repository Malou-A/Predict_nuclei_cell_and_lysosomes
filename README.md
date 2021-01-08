# Predict nuclei cell and lysosomes

## Description

This script is created to predict nuclei, cells and lysosomes from microscopy images. The input is expected to be a compressed .tar.gz folder containing the images.
The images are of C01 format and they end with either d0.C01, d1.C01 or d2.C01.

**The process in the script is as follows:**

Before iterating through the images (in pairs containing channel 0 and corresponding channel 1), all 3 models are loaded.

During iteration:
1. Extract the 2 images
2. convert the images from C01 to png
3. Normalize the images
4. Predict nuclei channel and retrieve information from the prediction (cell size, cell count, average cell intensity)
5. Predict cell channel and retrieve information from the prediction (total cell area, and calculate average cell area)
6. Predict cell channel with the lysosome model and retrieve lysosome information (count)
7. Predict bright parts of nuclei, using a pixel intensity threshold that is set with respect to the average nuclei intensity.
8. Add image information to an outputfile.

ex output of 3 image pairs:
```
Image	Average_nuclei_size	Nr_of_nuclei	Bright_parts	Average_nuclei_intensity	Average_cell_area	Lysosomes
MFGTMPcx7_170702090001_B23f06d0.png	1869.625	40	4	125.190	12727.1   15
MFGTMPcx7_170703220001_B23f11d0.png	1975.619	21	6	156.683	15889.8	  18
MFGTMPcx7_190816100002_B23f14d0.png	1683.812	80	7	101.153	10798.1 	51

```


## Dependencies


cuDNN/7.1.4.18
CUDA/9.2.88
Keras/2.2.4
TensorFlow/1.13.1
scikit-image/0.14.1
numpy/1.15.0



**bcftools and java:**

```bash
cd ~/bin
wget http://downloads.openmicroscopy.org/latest/bio-formats/artifacts/bftools.zip
unzip bftools.zip
rm bftools.zip
export PATH=$PATH:~/bin/bftools
```
Download and install java:

Download from https://www.java.com/en/download/linux_manual.jsp

(following commands are for the download for Linux x64)

```bash
cd ~/Downloads
tar -C ~/bin -zxvf jre-8u261-linux-x64.tar.gz
rm jre-8u261-linux-x64.tar.gz 
export PATH=$PATH:~/bin/jre-8u261-linux-x64/bin
```

