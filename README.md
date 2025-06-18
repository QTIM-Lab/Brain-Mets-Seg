# Brain-Mets-Seg

This repository contains code for automatic segmentation of brain metastases on T1-weighted contrast-enhanced MRI using a 3D U-Net architecture. The model was trained and validated on a large, multi-institutional dataset comprising 1,546 MR scans from 6 public and 2 private sources. It was evaluated on a held-out test cohort of 885 MR scans. In addition to segmentation, the repository includes tools for: 1) generating voxel-wise uncertainty maps, 2) performing longitudinal lesion tracking over time, 3) calculating automatic RANO-BM measurements, and 4) classifying treatment response based on the RANO criteria.

## Table of Contents

- [Model Overview](#model-overview)
- [Setup](#setup)
- [Preprocessing](#preprocessing)
- [Automatic Segmentation](#automatic-segmentation)
- [Longitudinal Tracking](#longitudinal-tracking)
- [Auto-RANO-BM](#auto-rano-bm)

## Model Overview

We use a nearly identical network architecture, optimizer setting, and data augmentation pipeline as nnU-Net, written in Tensorflow 2.10 and NVIDIA DALI. Specifically, we use a 6 level 3D U-Net which takes a T1-weighted contrast-enhanced (T1-CE) sequence image as input and outputs a probability map of the likely brain metastases. We use 32 filters in the 3x3x3 convolutions in the first layer, and double this number of filters as we go deeper into the network, capping the the total number of filters to 320 due to GPU memory constraints. Feature map downsampling and upsampling is accomplished through strided convolution and transposed convolution, respectively. Instance Normalization in lieu of Batch Normalization is used in order to accommodate the smaller batch size necessary to train a large patch 3D model. Leaky rectified linear unit (Leaky ReLU) activation was used in all layers, with the exception of the final linear output. And to encourage faster convergence and ensure that deeper layers of the decoder are learning semantically useful features, we employ deep supervision by integrating segmentation outputs from all but the two deepest levels of the network. Our model was trained with the SGD optimizer, with the unweighted summation of boundary-weighted cross-entropy and dice coefficient as the loss function.

A diagram of the architecture is shown below.

<img src="images/3D_UNet.png" width="900"/>

## Setup

The simplest way to use the trained model is via docker container. To do this, clone this repository and then build a docker image named `brain_mets_seg` using the provided Dockerfile:

```
git clone https://github.com/QTIM-Lab/Brain-Mets-Seg.git
docker build -t brain_mets_seg .
```

## Preprocessing

Before running segmentation, all data must be preprocessed. Our preprocessing pipeline is as follows: 1) resample all data to 1mm isotropic resolution, 2) N4 bias correction, 3) skull strip, 4) affine registration of all patient timepoints to the baseline scan, and 5) zero mean, unit variance intensity normalization (based on foreground voxels only). The preprocessing pipeline is currently located here: `https://github.com/QTIM-Lab/preprocessing`. Please follow the instructions on this github link to preprocess your data accordingly.

## Automatic Segmentation

To run the segmentation docker container on the preprocessed images (which should now be in NIFTI format (i.e. .nii.gz file ending)), you must create a csv file called `files_to_segment.csv`, which contains the full directory path to all images you wish to segment, and the names of those files. An example is shown below:

| Full_Patient_File_Path | T1_CE_Volume_Name |
| --- | --- |
| </full/file/path/to/image/one/> | <image_one_file_name> |
| </full/file/path/to/image/two/> | <image_two_file_name> |
| </full/file/path/to/image/three/> | <image_three_file_name> |
| ... | ... |

Once you have such a csv file, you can mount the directory with the csv (and any other data you need) to the docker container as follows:

```
docker run -it --privileged --runtime=nvidia --shm-size=8g --ulimit memlock=-1 --ulimit stack=67108864 --rm -v </full/path/to/csv/dir/>:/workspace/brain_mets_seg/data brain_mets_seg:latest /bin/bash
```

Once inside the docker container, you can segment the cases in your csv file as follows:

```
python predict.py
```

This will segment every scan using an ensemble of five models. This process will create three output files per segmentation. First, it will output an ensembled probability map called "model_ensemble_probability_map.nii.gz". Second, it will output an ensembled binary label map called "model_ensemble-label.nii.gz". And third, it will output a voxelwise uncertainty map called "model_ensemble_uncertainty_entropy_map.nii.gz".

## Longitudinal Tracking

If you have longitudinal patient data, you can also use the longitudinal tracking scripts in order to volumetrically track individual lesions across time. In order to use these scripts, your data must be organized in a certain way. Specifically, patient data should be organized such that all visits/timepoints for a patient are nested inside the main patient folder. An example of the required folder structure is shown below:

```
files_to_track
├── Patient_1/
│ ├── Visit_1/
│ │ └── model_ensemble-label.nii.gz
│ ├── Visit_2/
│ │ └── model_ensemble-label.nii.gz
│ └── Visit_3/
│   └── model_ensemble-label.nii.gz
├── Patient_2/
│ └── Visit_1/
│   └── model_ensemble-label.nii.gz
└── Patient_3/
  ├── Visit_1/
  │ └── model_ensemble-label.nii.gz
  └── Visit_2/
    └── model_ensemble-label.nii.gz
```

Once your data is organized like this, you can mount this data directory to your dockerin the same exact way as before. To run longitudinal tracking, you can use the following command from inside the docker:

```
python longitudinal_tracking.py
```

This will save out an additional label map called "model_ensemble_TRACKED-label.nii.gz", where every lesion is given a unique numerical identifier to allow it to be volumetrically tracked over time.

## Auto-RANO-BM

Once all patient imaging has been longitudinally tracked, you can use the Auto-RANO-BM script to calculate the RANO-BM measure on every timepoint, and categorize them as progressive disease (PD), stable disease (SD), partial response (PR), or complete response (CR) as outlined by the RANO-BM working group.

To do this, simply run the following command from inside the docker:

```
python calculate_rano_bm.py
```

This will output a csv file called "files_to_track_rano.csv", which will contain the number of lesions, the total tumor burden, the target tumor burden, the RANO-BM measure, and the response category.
