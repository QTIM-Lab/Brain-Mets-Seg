#CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1 python predict.py --data '/autofs/cluster/qtim/users/jn85/DeepLearningExamples/TensorFlow2/Segmentation/nnUNet/data/METS_Jay/_test_set_patients.csv' --fold 0
#CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1 python predict.py --fold 0
import os
import time
import numpy as np
import pandas as pd
import nibabel as nib
import tensorflow as tf

from glob import glob
from scipy.ndimage import find_objects
from data_loading.data_module import DataModule
from models.nn_unet import NNUnet
from runtime.args import get_main_args
from runtime.checkpoint import load_model
from runtime.logging import get_logger
from runtime.run import evaluate, export_model, predict, train
from runtime.utils import hvd_init, set_seed, set_tf_flags

def nested_folder_filepaths(nifti_dir, vols_to_process=None):
    if vols_to_process == None:
        relative_filepaths = [os.path.relpath(directory_paths, nifti_dir) for (directory_paths, directory_names, filenames) in os.walk(nifti_dir) if len(filenames)!=0]
    else:
        relative_filepaths = [os.path.relpath(directory_paths, nifti_dir) for (directory_paths, directory_names, filenames) in os.walk(nifti_dir) if all(vol_to_process in filenames if not isinstance(vol_to_process, list) else any(vol_to_process2 in filenames for vol_to_process2 in vol_to_process) for vol_to_process in vols_to_process)]
    return relative_filepaths

def predict_on_patient(full_patient_path, vol_name, fold, patch_size, index_patient, num_patients, save_logits=True, save_binary_label=True, logits_file_name='logits_fold', binary_label_file_name='model_fold', timer=True):
    if timer == True:
        start_prediction_time = time.time()
    #
    nib_vol = nib.load(full_patient_path + vol_name)
    T1_im = nib_vol.get_fdata()
    #crop to nonzero foreground
    if np.any(T1_im != 0):
        volume_location = find_objects(T1_im != 0)[0]
    else:
        volume_location = find_objects(T1_im == 0)[0]
    T1_im_crop = T1_im[volume_location]
    #pad all axes to the patch size in case cropped volume is smaller than size of patch along an axis or axes
    extra_padding = np.maximum(patch_size - np.array(T1_im_crop.shape), 0)
    pad_tuple_initial = tuple([(int(np.floor(i / 2)), int(np.ceil(i / 2))) for i in extra_padding])
    T1_im_crop = np.pad(T1_im_crop, pad_tuple_initial, mode='constant')
    #
    x = tf.cast(tf.constant(np.expand_dims(T1_im_crop, axis=(0,-1))), tf.float32)
    y = model.inference(x)[0,...,0].numpy()
    #
    remove_padding_initial_index = tuple([slice(start_index, y.shape[i] - end_index, 1) for i, (start_index, end_index) in enumerate(pad_tuple_initial)])
    y = y[remove_padding_initial_index]
    pad_bounds = [(j.start, T1_im.shape[i]-j.stop) for i,j in enumerate(volume_location)]
    y = np.pad(y, pad_bounds, mode='constant')
    #
    if save_logits == True:
        nib_vol_save = nib.Nifti1Image(y, affine=nib_vol.affine, header=nib_vol.header)
        nib.save(nib_vol_save, full_patient_path + logits_file_name + fold + '.nii.gz')
    if save_binary_label == True:
        nib_vol_save = nib.Nifti1Image((y > 0).astype(int), affine=nib_vol.affine, header=nib_vol.header)
        nib.save(nib_vol_save, full_patient_path + binary_label_file_name + fold + '-label.nii.gz')
    #
    if timer == True:
        time_for_prediction = np.round(time.time() - start_prediction_time, 2)
        print(str(index_patient+1) + '/' + str(num_patients) + ': Prediction for patient ' + full_patient_path + ' is complete (' + str(time_for_prediction) + 's) \n')

if __name__ == "__main__":
    args = get_main_args()
    args.exec_mode = 'predict'
    args.gpus = 1
    args.dim = 3
    args.seed = 12345
    args.amp = True
    args.xla = True
    args.batch_size = 2
    args.steps_per_epoch = 250
    args.epochs = 1000
    args.skip_eval = 0
    args.skip_train_eval = 0
    args.optimizer = 'sgdw' 
    args.learning_rate = 0.01
    args.end_learning_rate = 0.0
    args.weight_decay = 0.00003
    args.momentum = 0.99
    args.dampening = 0.0
    args.nesterov = True
    args.decoupled = False
    args.scheduler = 'poly'
    args.power = 0.9
    args.deep_supervision = True
    args.loss_batch_reduction = False
    args.overlap = 0.75 
    args.tta = True
    args.save_preds = True
    args.blend_mode = 'gaussian'
    args.negative_slope = 0.01 
    args.use_hvd = False
    args.results = '/autofs/cluster/qtim/users/jn85/DeepLearningExamples/TensorFlow2/Segmentation/nnUNet/results_mets_fold' + str(args.fold) + '_dice_bound_ce/'
    args.ckpt_dir = '/autofs/cluster/qtim/users/jn85/DeepLearningExamples/TensorFlow2/Segmentation/nnUNet/results_mets_fold' + str(args.fold) + '_dice_bound_ce/ckpt/best/'
    hvd_init(args.use_hvd)
    if args.seed is not None:
        set_seed(args.seed)
    set_tf_flags(args)
    model = load_model(args)
    #
    '''
    #DIRECTLY PREDICT ON ONE IMAGE
    patient_path = '/autofs/cluster/qtim/datasets/private/METS_Jay/BRATS_BrainMets/ASNR-MICCAI-BraTS2023-MET-Challenge-TrainingData/BraTS-MET-00406-000/'
    os.chdir(patient_path)
    nib_vol = nib.load('T1C_RAI_RESAMPLED_N4_SS_REG_NORM.nii.gz')
    x = tf.cast(tf.constant(np.expand_dims(nib_vol.get_fdata(), axis=(0,-1))), tf.float32)
    y = model.inference(x)
    nib_vol_save = nib.Nifti1Image((y[0,...,0].numpy() > 0).astype(int), affine=nib_vol.affine, header=nib_vol.header)
    nib.save(nib_vol_save, 'model_nnunet-label.nii.gz')
    '''
    #
    #PREDICT ON IMAGES FROM CSV
    vol_name = 'T1C_RAI_RESAMPLED_N4_SS_REG_NORM.nii.gz'
    patch_size = np.array([128]*3)
    df_csv = pd.read_csv(args.data, index_col=0)
    patients = list(df_csv['Test_Set_Patient_File_Path'])
    num_patients = len(patients)
    for i, patient in enumerate(patients):
        predict_on_patient(patient + '/', vol_name, str(args.fold), patch_size, i, num_patients)
    #
    '''
    #PREDICT ON FILE DIRECTORY
    vol_name = 'T1C_RAI_RESAMPLED_N4_SS_REG_NORM.nii.gz'
    patch_size = np.array([128]*3)
    folder_dir = '/autofs/cluster/qtim/datasets/private/METS_Jay/Ray_set1_nii_new_BrainMets/'
    patients = sorted(nested_folder_filepaths(folder_dir, vols_to_process=[vol_name]))
    num_patients = len(patients)
    for i, patient in enumerate(patients):
        predict_on_patient(folder_dir + patient + '/', vol_name, str(args.fold), patch_size, i, num_patients)
    '''